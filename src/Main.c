#include<stdio.h>
#include "network/Network.h"
#include "cli/NetworkCLI.h"
#include <stdlib.h>
#include <string.h>
#include<math.h>


//CONSTANT SETUP
	void saveErrorInfo(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients) {
		NetworkLayer *out = net->layers + net->layerCount - 1;
		unsigned short int len = out->neuronCount;

		NeuronUnit sum = 0;
		for (unsigned short int i=len; i--;) {
			errorGradients[i] = 2*(out->neurons[i].out - expected[i]);
			sum += errorGradients[i] * errorGradients[i] / 4;
		}

		*errorValue = sum;
	}

	void addToErrorInfo(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients) {
		NetworkLayer *out = net->layers + net->layerCount - 1;
		unsigned short int len = out->neuronCount;

		NeuronUnit sum = 0;
		for (unsigned short int i=len; i--;) {
			NeuronUnit toBeAdded = 2*(out->neurons[i].out - expected[i]);
			errorGradients[i] += toBeAdded;
			sum += toBeAdded * toBeAdded / 4;
		}

		*errorValue += sum;
	}



//STRUCTURE
	/** This function feeds the next input, and the expected output. */
	char getNextInput(TrainDataProvider* provider, NeuralNetwork* net) {
		provider->counter++;
		if (provider->counter > provider->maxResults) return 0;


		NetworkLayer *inp = net->layers;
		char input1 = (NeuronUnit)rand() / ((NeuronUnit)RAND_MAX) > 0.5? 1 : 0 ;
		char input2 = (NeuronUnit)rand() / ((NeuronUnit)RAND_MAX) > 0.5? 1 : 0 ;

		inp->neurons[0].out = input1;
		inp->neurons[1].out = input2;
		provider->expected[0] = input1 ^ input2;
		return 1;
	}


	void startCLI() {
		NeuralNetworkStructure str = {
			(NetworkLayerStructure[]) {
				//INPUT LAYER
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_FULLY_CONNECTED,
					.neuronCount = 2,
				},

				//HIDDEN LAYER 1
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_INDIVIDUAL,
					.neurons = (int*[]) {
						(int[]) {1, 2, -1},
						(int[]) {0, 1, -1},
						(int[]) {0, 2, -1},
						(int[]) {0, 2, -1},
						(int[]) {0, 1, 2, -1},
						NULL
					},
					.bias = (int[]) {-1},
					.activatorType = NeuronActivator_LEAKY_RELU
				},
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_FULLY_CONNECTED,
					.neuronCount = 3,
					.activatorType = NeuronActivator_LEAKY_RELU
				},

				//OUTPUT LAYER
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_OUTPUT,
					.neuronCount = 1,
					.activatorType = NeuronActivator_LINEAR,
				},
			}
		};


		NeuralNetwork net;
		TrainDataProvider provider;

		NeuralNetwork_init(&net, &str);
		NeuralNetwork_randomSynapses(&net);
		TrainDataProvider_init(&provider, *getNextInput, net.layers[net.layerCount - 1].neuronCount, 10000);

		NetworkCLI_start(&net, &provider, *saveErrorInfo, *saveErrorInfo);

		//clean up
		NeuralNetwork_deinit(&net);
		TrainDataProvider_deinit(&provider);
	}




	#ifndef UNIT_TESTS
		int main(int argc, char** argv) {
			startCLI();
			return 0;
		}
	#endif
