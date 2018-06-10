#include "NetworkTrain.h"
#include <stdlib.h>

void BPTrainer_init(
		BPTrainer* this,
		NeuralNetwork* network,
		TrainDataProvider *provider,
		void (*errorUpdater)(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients)) {
	this->network = network;
	this->isTraining = 0;

	this->start.weights = (NeuronUnit*) malloc(network->synapseCount * sizeof(NeuronUnit));
	this->minimum.weights = (NeuronUnit*) malloc(network->synapseCount * sizeof(NeuronUnit));
	this->minimum.error = 999;

	this->provider = provider;
	this->errorUpdater = errorUpdater;
}

void BPTrainer_deinit(BPTrainer* this) {
	free(this->minimum.weights);
	free(this->start.weights);
}


//UTILS
	static void zeroOut(NeuronUnit* target, unsigned int length) {
		for (unsigned int i = length - 1; 1; --i) {
			target[i] = 0;
			if (i==0) break;
		}
	}

	static void printDebugInfo(BPTrainer* this, NeuronUnit errorValue) {
		NetworkLayer *inp = this->network->layers;
		NetworkLayer *out = this->network->layers + this->network->layerCount - 1;
		NeuronUnit *expected = this->provider->expected;
		unsigned short int inpCount = inp->neuronCount;
		unsigned short int outCount = out->neuronCount;

		printf("INPUT:");
		for (unsigned short int i = 0; i <inpCount; ++i) printf(" %1.2f", inp->neurons[i].out);
		printf("\nOUTPUT:");
		for (unsigned short int i = 0; i <outCount; ++i) printf(" %1.2f", out->neurons[i].out);
		printf("\nEXPECTED:");
		for (unsigned short int i = 0; i <outCount; ++i) printf(" %1.2f", expected[i]);
		printf("\nERROR: %1.2f\n\n", errorValue);
	}



//ONLINE TRAINING
	void BPTrainer_trainOnline(BPTrainer* this, NeuronUnit learningRate, NeuronUnit momentum, char debug) {
		NeuralNetwork *network = this->network;
		NeuronUnit* adjustment2 = malloc(network->synapseCount * sizeof(NeuronUnit));
		NeuronUnit* adjustment1 = malloc(network->synapseCount * sizeof(NeuronUnit));
		NeuronUnit* errorDerivatives = malloc(network->synapseCount * sizeof(NeuronUnit));

		NeuronUnit* lastAdjustment = adjustment1;
		NeuronUnit errorValue=0;

		zeroOut(errorDerivatives, network->synapseCount);
		zeroOut(adjustment1, network->synapseCount);
		zeroOut(adjustment2, network->synapseCount);

		TrainDataProvider* provider = this->provider;
		char (*provideFunc)(TrainDataProvider*, NeuralNetwork*) = provider->provideInput;

		this->isTraining = 1;
		while(this->isTraining) {
			char hasNext = provideFunc(provider, network);
			if (hasNext==0) break;

			//see current error
			NeuralNetwork_predict(network);
			this->errorUpdater(this->network, provider->expected, &errorValue, errorDerivatives);
			if (debug) printDebugInfo(this, errorValue);


			//is it a new minimum? If so, save it.
			if (errorValue < this->minimum.error) {
				this->minimum.error = errorValue;
				printf("Minimum error found: %f\n", errorValue);
				NeuralNetwork_saveSynapseWeights(network, this->minimum.weights);
			}

			//calculate adjustment
			NeuronUnit *currentAdjustment = (lastAdjustment == adjustment1)? adjustment2 : adjustment1;
			NeuralNetwork_saveGradient(network, errorDerivatives, currentAdjustment);

			for (unsigned int i = network->synapseCount; i--;) {
				currentAdjustment[i] = - currentAdjustment[i]*learningRate + lastAdjustment[i]*momentum;
			}

			NeuralNetwork_adjustWeights(network, currentAdjustment);
			lastAdjustment = currentAdjustment;
		}

		//clean up
		free(errorDerivatives);
		free(adjustment1);
		free(adjustment2);
	}

//STOCHASTIC TRAINING
	void BPTrainer_trainStochastic(BPTrainer* this, unsigned int updateEvery, NeuronUnit learningRate, NeuronUnit momentum, char debug) {
		NeuralNetwork *network = this->network;
		NetworkLayer *outputLayer = network->layers + network->layerCount-1;

		NeuronUnit* adjustment2 = malloc(network->synapseCount * sizeof(NeuronUnit));
		NeuronUnit* adjustment1 = malloc(network->synapseCount * sizeof(NeuronUnit));
		NeuronUnit* currentErrorDerivatives = malloc(outputLayer->synapseCount * sizeof(NeuronUnit));

		NeuronUnit *lastAdjustment = adjustment1,
					*currentAdjustment = adjustment2;
		NeuronUnit currentErrorValue, errorValueSum = 0;;

		zeroOut(currentErrorDerivatives, network->synapseCount);
		zeroOut(adjustment1, network->synapseCount);
		zeroOut(adjustment2, network->synapseCount);

		TrainDataProvider* provider = this->provider;
		char (*provideFunc)(TrainDataProvider*, NeuralNetwork*) = provider->provideInput;
		unsigned int counter = 1;

		this->isTraining = 1;
		while(this->isTraining) {
			char hasNext = provideFunc(provider, network);
			if (hasNext==0) break;

			//see current error
			NeuralNetwork_predict(network);
			this->errorUpdater(this->network, provider->expected, &currentErrorValue, currentErrorDerivatives);
			errorValueSum += currentErrorValue;
			if (debug) printDebugInfo(this, currentErrorValue);

			//calculate adjustment
			NeuralNetwork_addToGradient(network, currentErrorDerivatives, currentAdjustment);

			/*printf("Input:\n  ");
			for (int i=0; i<network->layers[0].neuronCount; ++i) printf("%1.2f ", network->layers[0].neurons[i].out);
			printf("Expected: %1.2f     FOUND: %1.2f\n", provider->expected[0], network->layers[network->layerCount-1].neurons[0].out);
			printf("\ngradients:\n  ");
			for (int i=0; i<network->synapseCount; ++i) printf("%1.2f ", currentAdjustment[i]);
			printf("\n\n");*/

			//update
			if (counter >= updateEvery) {
				errorValueSum /= counter; //Average error
				//printf("Average error:  %1.2f  (%d samples) \n", errorValueSum, counter);
				if (errorValueSum < this->minimum.error) {
					this->minimum.error = errorValueSum;
					printf("Minimum error found: %f\n", errorValueSum);
					NeuralNetwork_saveSynapseWeights(network, this->minimum.weights);
				}

				for (unsigned int i = network->synapseCount; i--;) {
					currentAdjustment[i] =  - (currentAdjustment[i]/counter)*learningRate + lastAdjustment[i]*momentum;
				}

				NeuralNetwork_adjustWeights(network, currentAdjustment);

				//swap adjustments and resrtart counter
				lastAdjustment = currentAdjustment;
				currentAdjustment = (lastAdjustment == adjustment1)? adjustment2 : adjustment1;

				//zero out errors
				zeroOut(currentAdjustment, network->synapseCount);
				errorValueSum = 0;
				counter=1;
				//printf("##############################\n");
			}
			else counter++;
		}

		//clean up
		free(currentErrorDerivatives);
		free(adjustment1);
		free(adjustment2);
	}
