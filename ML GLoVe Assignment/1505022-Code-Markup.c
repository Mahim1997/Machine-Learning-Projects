/* 	Relevant codes from the codebase:  https://github.com/stanfordnlp/GloVe 
	Code File: https://github.com/stanfordnlp/GloVe/blob/master/src/glove.c
*/

void initialize_parameters() {
	// Paramters for W are initialized here
	for (a = 0; a < W_size; ++a) {
		W[a] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
	}

	// Squared gradients are initialized here
	for (a = 0; a < W_size; ++a) {
		gradsq[a] = 1.0; // So initial value of eta is equal to initial learning rate
	}
}

void *glove_thread(void *vid) { /* Train the GloVe model (single function) */
    
    cost[id] = 0;
    
    real* W_updates1 = (real*)malloc(vector_size * sizeof(real)); // updates of weights in this variable
    real* W_updates2 = (real*)malloc(vector_size * sizeof(real)); // updates of weights in this variable

    for (a = 0; a < lines_per_thread[id]; a++) {
        /* Get location of words in W & gradsq */
        l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words
        
        /* Calculate cost, save diff for gradients */
        diff = 0;
        for (b = 0; b < vector_size; b++) diff += W[b + l1] * W[b + l2]; // dot product of word and context word vector
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val); // add separate bias for each word
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff

        cost[id] += 0.5 * fdiff * diff; // weighted squared error  
        real W_updates1_sum = W_updates2_sum = 0; /* Adaptive gradient updates */

        for (b = 0; b < vector_size; b++) {
            
            temp1 = fmin(fmax(fdiff * W[b + l2], -grad_clip_value), grad_clip_value) * eta; // learning rate times gradient for word vectors
            temp2 = fmin(fmax(fdiff * W[b + l1], -grad_clip_value), grad_clip_value) * eta; // learning rate times gradient for word vectors
            
			// adaptive updates for AdaGrad
            W_updates1[b] = temp1 / sqrt(gradsq[b + l1]);
            W_updates2[b] = temp2 / sqrt(gradsq[b + l2]);
            W_updates1_sum += W_updates1[b];
            W_updates2_sum += W_updates2[b];
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;
        }
		//Updates for AdaGrad
		for (b = 0; b < vector_size; b++) {
			W[b + l1] -= W_updates1[b];
			W[b + l2] -= W_updates2[b];
		}

        // Updates for bias
        W[vector_size + l1] -= check_nan(fdiff / sqrt(gradsq[vector_size + l1]));
        W[vector_size + l2] -= check_nan(fdiff / sqrt(gradsq[vector_size + l2]));
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
        
    }
}
int train_glove() { /* Train model using multiple threads */
    for (b = 0; b < num_iter; b++) { // Lock-free asynchronous SGD
        total_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        long long *thread_ids = (long long*)malloc(sizeof(long long) * num_threads); // Allocate memory for threads
        for (a = 0; a < num_threads; a++) thread_ids[a] = a; // Create threads
        // Call the 'glove_thread' single thread training function
		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)&thread_ids[a]); 
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL); // Join each thread
        for (a = 0; a < num_threads; a++) total_cost += cost[a]; // Update total cost
        free(thread_ids);
    }
    free(lines_per_thread); // Delete the threads
    return save_params(-1);
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    int result = 0;
            
	vocab_size = 0;
	fid = fopen(vocab_file, "r"); // Read in vocabulary file
	if (fid == NULL) {log_file_loading_error("vocab file", vocab_file); free(cost); return 1;}
	while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
	fclose(fid);
	if (vocab_size == 0) {fprintf(stderr, "Unable to find any vocab entries in vocab file %s.\n", vocab_file); free(cost); return 1;}
	result = train_glove();

    free(W);
    free(gradsq);

    return result;
}