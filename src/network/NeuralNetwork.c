#include "Network.h"
#include <stdlib.h>
#include <time.h>
#include<math.h>



//LIFE CIRCLE
	void NeuralNetwork_init(NeuralNetwork* this, NeuralNetworkStructure* s) {
		unsigned short int layerCount = 0;
		while(s->layers[layerCount].connectionType != NetworkLayer_OUTPUT) layerCount++;
		layerCount++;
		this->layerCount = layerCount;

		this->layers = (NetworkLayer*) malloc(layerCount * sizeof(NetworkLayer));

		//setup layer structure
		unsigned short int neuronCount = 0;
		unsigned long int synapseCount = 0;
		if (layerCount > 0) {
			for (unsigned short int i = layerCount; i--;) {
				NetworkLayer *currentLayer = this->layers + i;
				NetworkLayerStructure *currentLayerStr = s->layers + i;
				NetworkLayerStructure *nextLayerStr;

				switch (currentLayerStr->connectionType) {
					case NetworkLayer_FULLY_CONNECTED:
						nextLayerStr = s->layers + i + 1;
						NetworkLayer_initFullyConnected(currentLayer, currentLayerStr, NetworkLayer_getNeuronCount(nextLayerStr));
						break;

					case NetworkLayer_INDIVIDUAL:
						NetworkLayer_initIndividual(currentLayer, currentLayerStr);
						break;

					default:
						NetworkLayer_initOutput(currentLayer, currentLayerStr);
				}
				neuronCount += currentLayer->neuronCount; //+1 for the bias
				synapseCount += currentLayer->synapseCount;
			}
		}
		this->neuronCount = neuronCount;
		this->synapseCount = synapseCount;


		//connect layers
		if (layerCount > 1) {
			for (unsigned short int i = layerCount -1; 1; --i) {
				NetworkLayer_bindForward(&(this->layers[i-1]), &(this->layers[i]));
				if (i<=1) break;
			}
		}
	}

	void NeuralNetwork_deinit(NeuralNetwork* this) {
		for (unsigned short int i = this->layerCount; i--;) NetworkLayer_deinit(this->layers + i);
		free(this->layers);
	}




//STATE SETUP
	void NeuralNetwork_randomSynapses(NeuralNetwork* this) {
		if (this->layerCount <= 1) return;
		srand(time(NULL));
		for (unsigned short int i = this->layerCount-1; i--;) NetworkLayer_randomSynapses(this->layers + i);
	}

	void NeuralNetwork_saveSynapseWeights(NeuralNetwork* this, NeuronUnit* buffer) {
		if (this->layerCount <= 1) return;

		NeuronUnit* layerWeights = buffer + this->synapseCount;
		for (unsigned short int i = this->layerCount-1; i--;) {
			NetworkLayer *current = this->layers + i;
			layerWeights -= current->synapseCount;
			NetworkLayer_saveSynapseWeights(current, layerWeights);
		}
	}

	void NeuralNetwork_loadSynapseWeights(NeuralNetwork* this, NeuronUnit* weights) {
		if (this->layerCount <= 1) return;

		NeuronUnit* layerWeights = weights + this->synapseCount;
		for (unsigned short int i = this->layerCount-1; i--;) {
			NetworkLayer *current = this->layers + i;
			layerWeights -= current->synapseCount;
			NetworkLayer_loadSynapseWeights(current, layerWeights);
		}
	}

	void NeuralNetwork_adjustWeights(NeuralNetwork* this, NeuronUnit* offsets) {
		if (this->layerCount <= 1) return;

		NeuronUnit* layerWeights = offsets + this->synapseCount;
		for (unsigned short int i = this->layerCount-1; i--;) {
			NetworkLayer *current = this->layers + i;
			layerWeights -= current->synapseCount;
			NetworkLayer_adjustWeights(current, layerWeights);
		}
	}



//NETWORK OPERATIONS
	void NeuralNetwork_predict(NeuralNetwork * this) {
		if (this->layerCount <= 1) return;

		for (unsigned short int i = 0, len = this->layerCount-1; i<len; ++i) {
			NetworkLayer* current = this->layers + i;
			NetworkLayer* next = current + 1;
			NetworkLayer_reset(next);
			NetworkLayer_fire(current);
			NetworkLayer_activate(next);
		}
	}

	void NeuralNetwork_saveGradient(NeuralNetwork *this, NeuronUnit *errorDerivatives, NeuronUnit* grad) {
		if (this->layerCount <= 1) return;

		//calculate deltas for last layer (no gradients)
		unsigned short int currentLayerIndex = this->layerCount - 1;
		NetworkLayer_calculateDeltaFromErrorDerivatives(this->layers + currentLayerIndex, errorDerivatives);

		//calculate gradients and deltas for each other layer, exept the first.
		NeuronUnit* layerGrad = grad + this->synapseCount;
		while(currentLayerIndex-- > 1) {
			NetworkLayer* currentLayer = this->layers + currentLayerIndex;
			layerGrad -= currentLayer->synapseCount;
			NetworkLayer_saveGradient(currentLayer, layerGrad); //NULL indicates that we dont care about delta calculation.
		}

		//calculate gradients for the first layer, (NO deltas)
		NetworkLayer* inputLayer = this->layers;
		layerGrad -= inputLayer->synapseCount;
		NetworkLayer_saveGradient(inputLayer, layerGrad); //NULL indicates that we dont care about delta calculation.
	}

	void NeuralNetwork_addToGradient(NeuralNetwork *this, NeuronUnit *errorDerivatives, NeuronUnit* grad) {
		if (this->layerCount <= 1) return;

		//calculate deltas for last layer (no gradients)
		unsigned short int currentLayerIndex = this->layerCount - 1;
		NetworkLayer_calculateDeltaFromErrorDerivatives(this->layers + currentLayerIndex, errorDerivatives);

		//calculate gradients and deltas for each other layer, exept the first.
		NeuronUnit* layerGrad = grad + this->synapseCount;
		while(currentLayerIndex-- > 1) {
			NetworkLayer* currentLayer = this->layers + currentLayerIndex;
			layerGrad -= currentLayer->synapseCount;
			NetworkLayer_addToGradient(currentLayer, layerGrad); //NULL indicates that we dont care about delta calculation.
		}

		//calculate gradients for the first layer, (NO deltas)
		NetworkLayer* inputLayer = this->layers;
		layerGrad -= inputLayer->synapseCount;
		NetworkLayer_addToGradient(inputLayer, layerGrad); //NULL indicates that we dont care about delta calculation.
	}
