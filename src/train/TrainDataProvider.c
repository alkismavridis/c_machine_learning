#include "NetworkTrain.h"
#include <stdlib.h>


void TrainDataProvider_init(
		TrainDataProvider* this,
		char (*provideInput)(TrainDataProvider* this, NeuralNetwork* net),
		unsigned short int outputCount,
		unsigned int maxResults) {
	this->provideInput = provideInput;
	this->counter = 0;
	this->maxResults = maxResults;
	this->expected = malloc(outputCount * sizeof(NeuronUnit));
}

void TrainDataProvider_deinit(TrainDataProvider* this) {
	free(this->expected);
}

void TrainDataProvider_reset(TrainDataProvider* this, unsigned int maxResults) {
	this->counter = 0;
	this->maxResults = maxResults;
}
