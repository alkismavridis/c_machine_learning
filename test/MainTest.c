#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "../src/network/Network.h"



//assertion function
	typedef struct {
		char* name;
	} TestCase;

	void assertIntEqual(long int expected, long int actual, TestCase *t, char* label) {
		if (expected != actual) {
			printf("%s %s\n\tExpected %lu,    but found %lu instead.\n", t->name, label, expected, actual);
			exit(1);
		}
	}

	void assertDoubleEqual(double expected, double actual, double tolerance, TestCase *t, char* label) {
		if (fabs(expected - actual) > tolerance) {
			printf("%s %s\n\tExpected %f,    but found %f instead.\n", t->name, label, expected, actual);
			exit(1);
		}
	}

	void assertPtrEqual(void* expected, void* actual, TestCase *t, char* label) {
		if (expected != actual) {
			printf("%s %s\n\tExpected %p,    but found %p instead.\n", t->name, label, expected, actual);
			exit(1);
		}
	}



//UTILS FUNCTIONS
	NeuronUnit dummyActivation(NeuronUnit x) { return x*x/4 + x; }
	NeuronUnit dummyActivationDerivative(NeuronUnit x) { return x/2 + 1; }


	void createSimpleStructure(NetworkLayer *layer1, NetworkLayer *layer2) {
		NetworkLayerStructure str1 = {
			.connectionType = NetworkLayer_INDIVIDUAL,
			.neurons = (int*[]) {
				(int[]) {0, 2, -1},
				(int[]) {0, 1, 2, -1},
				NULL
			},
			.bias = (int[]) {0, 1, 2, -1},
			.activatorType = NeuronActivator_SIGMOID
		};

		NetworkLayerStructure str2 = {
			.connectionType = NetworkLayer_INDIVIDUAL,
			.neurons = (int*[]) {
				(int[]) {-1},
				(int[]) {-1},
				(int[]) {-1},
				NULL
			},
			.bias = (int[]) {-1},
			.activatorType = NeuronActivator_LINEAR
		};

		NetworkLayer_initIndividual(layer1, &str1);
		NetworkLayer_initIndividual(layer2, &str2);

		//dummy input
		layer1->neurons[0].out = 0.4;
		layer1->neurons[1].out = 0.6;

		//adjust weights
		layer1->neurons[0].synapses[0].weight = 0.1;
		layer1->neurons[0].synapses[1].weight = -0.25;

		layer1->neurons[1].synapses[0].weight = 0.2;
		layer1->neurons[1].synapses[1].weight = 0.5;
		layer1->neurons[1].synapses[2].weight = 0.1;

		layer1->bias.synapses[0].weight = -0.3;
		layer1->bias.synapses[1].weight = -0.1;
		layer1->bias.synapses[2].weight = 0.9;

		//connect layers
		NetworkLayer_bindForward(layer1, layer2);
	}


	void createSimpleNetwork(NeuralNetwork *net) {
		NeuralNetworkStructure str = {
			(NetworkLayerStructure[]) {
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_INDIVIDUAL,
					.neurons = (int*[]) {
						(int[]) {0, 2, -1},
						(int[]) {0, 1, 2, -1},
						NULL
					},
					.bias = (int[]) {0, 1, 2, -1},
				},
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_INDIVIDUAL,
					.neurons = (int*[]) {
						(int[]) {0, 1, -1},
						(int[]) {0, 1, -1},
						(int[]) {1, -1},
						NULL
					},
					.bias = (int[]) {0, 1, -1},
					.activatorType = NeuronActivator_SIGMOID
				},
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_FULLY_CONNECTED,
					.neuronCount = 2,
					.activatorType = NeuronActivator_LINEAR
				},
				(NetworkLayerStructure) {
					.connectionType = NetworkLayer_OUTPUT,
					.neuronCount = 2,
					.activatorType = NeuronActivator_SIGMOID
				}
			}
		};

		NeuralNetwork_init(net, &str);

		//dummy input
		NetworkLayer * inputLayer = &(net->layers[0]);
		inputLayer->neurons[0].out = 0.4;
		inputLayer->neurons[1].out = 0.6;

		//set weights of inputLayer
		inputLayer->neurons[0].synapses[0].weight = 0.1;
		inputLayer->neurons[0].synapses[1].weight = -0.25;

		inputLayer->neurons[1].synapses[0].weight = 0.2;
		inputLayer->neurons[1].synapses[1].weight = 0.5;
		inputLayer->neurons[1].synapses[2].weight = 0.1;

		inputLayer->bias.synapses[0].weight = -0.3;
		inputLayer->bias.synapses[1].weight = -0.1;
		inputLayer->bias.synapses[2].weight = 0.9;


		//adjust weights of first hidden layer 1
		NetworkLayer * hidden1 = &(net->layers[1]);
		hidden1->neurons[0].synapses[0].weight = 0.5;
		hidden1->neurons[0].synapses[1].weight = 0.2;
		hidden1->neurons[1].synapses[0].weight = -0.9;
		hidden1->neurons[1].synapses[1].weight = -0.4;
		hidden1->neurons[2].synapses[0].weight = 0.3;
		hidden1->bias.synapses[0].weight = 0.1;
		hidden1->bias.synapses[1].weight = -0.2;

		//adjust weights of first hidden layer 2
		NetworkLayer * hidden2 = &(net->layers[2]);
		hidden2->neurons[0].synapses[0].weight = 0.2;
		hidden2->neurons[0].synapses[1].weight = 0.9;
		hidden2->neurons[1].synapses[0].weight = -0.6;
		hidden2->neurons[1].synapses[1].weight = -0.3;
		hidden2->bias.synapses[0].weight = -0.2;
		hidden2->bias.synapses[1].weight = 0.8;
	}





//NETWORK TESTS
	void testNetworkInit(TestCase *t) {
		NeuralNetwork net;
		createSimpleNetwork(&net);

		//test Network counters
		assertIntEqual(4, net.layerCount, t, "pre1");
		assertIntEqual(2+3+2+2, net.neuronCount, t, "pre2");
		assertIntEqual(2+3+3 + 2+2+1+2 + 2+2+2, net.synapseCount, t, "pre3");


		//test layer counters
		assertIntEqual(2, net.layers[0].neuronCount, t, "A1");
		assertIntEqual(3, net.layers[1].neuronCount, t, "A2");
		assertIntEqual(2, net.layers[2].neuronCount, t, "A3");
		assertIntEqual(2, net.layers[3].neuronCount, t, "A4");

		assertIntEqual(8, net.layers[0].synapseCount, t, "A5");
		assertIntEqual(7, net.layers[1].synapseCount, t, "A6");
		assertIntEqual(6, net.layers[2].synapseCount, t, "A7");
		assertIntEqual(0, net.layers[3].synapseCount, t, "A8");

		//test neuron counters
		assertIntEqual(2, net.layers[0].neurons[0].synapseCount, t, "B1");
		assertIntEqual(3, net.layers[0].neurons[1].synapseCount, t, "B2");
		assertIntEqual(3, net.layers[0].bias.synapseCount, t, "B3");

		assertIntEqual(2, net.layers[1].neurons[0].synapseCount, t, "B4");
		assertIntEqual(2, net.layers[1].neurons[1].synapseCount, t, "B5");
		assertIntEqual(1, net.layers[1].neurons[2].synapseCount, t, "B6");
		assertIntEqual(2, net.layers[1].bias.synapseCount, t, "B7");

		assertIntEqual(2, net.layers[2].neurons[0].synapseCount, t, "B8");
		assertIntEqual(2, net.layers[2].neurons[1].synapseCount, t, "B9");
		assertIntEqual(2, net.layers[2].bias.synapseCount, t, "B10");

		assertIntEqual(0, net.layers[3].neurons[0].synapseCount, t, "B11");
		assertIntEqual(0, net.layers[3].neurons[1].synapseCount, t, "B12");
		assertIntEqual(0, net.layers[3].bias.synapseCount, t, "B13");

		//test connections
		assertPtrEqual(&(net.layers[1].neurons[0]), net.layers[0].neurons[0].synapses[0].target, t, "C1");
		assertPtrEqual(&(net.layers[1].neurons[2]), net.layers[0].neurons[0].synapses[1].target,  t, "C2");
		assertPtrEqual(&(net.layers[1].neurons[0]), net.layers[0].neurons[1].synapses[0].target,  t, "C3");
		assertPtrEqual(&(net.layers[1].neurons[1]), net.layers[0].neurons[1].synapses[1].target,  t, "C4");
		assertPtrEqual(&(net.layers[1].neurons[2]), net.layers[0].neurons[1].synapses[2].target,  t, "C5");
		assertPtrEqual(&(net.layers[1].neurons[0]), net.layers[0].bias.synapses[0].target,  t, "C6");
		assertPtrEqual(&(net.layers[1].neurons[1]), net.layers[0].bias.synapses[1].target,  t, "C7");
		assertPtrEqual(&(net.layers[1].neurons[2]), net.layers[0].bias.synapses[2].target,  t, "C8");

		assertPtrEqual(&(net.layers[2].neurons[0]), net.layers[1].neurons[0].synapses[0].target, t, "C9");
		assertPtrEqual(&(net.layers[2].neurons[1]), net.layers[1].neurons[0].synapses[1].target,  t, "C10");
		assertPtrEqual(&(net.layers[2].neurons[0]), net.layers[1].neurons[1].synapses[0].target,  t, "C11");
		assertPtrEqual(&(net.layers[2].neurons[1]), net.layers[1].neurons[1].synapses[1].target,  t, "C12");
		assertPtrEqual(&(net.layers[2].neurons[1]), net.layers[1].neurons[2].synapses[0].target,  t, "C13");
		assertPtrEqual(&(net.layers[2].neurons[0]), net.layers[1].bias.synapses[0].target,  t, "C14");
		assertPtrEqual(&(net.layers[2].neurons[1]), net.layers[1].bias.synapses[1].target,  t, "C15");

		NeuralNetwork_deinit(&net);
	}

	void testNetworkWeightsManagement(TestCase *t) {
		NeuralNetwork net;
		createSimpleNetwork(&net);
		NeuronUnit* buf = malloc(net.synapseCount * sizeof(NeuronUnit) + 1);
		buf[net.synapseCount] = 0; //test for bound violations. this must stay 0

		srand(time(NULL));
		NeuralNetwork_randomSynapses(&net);
		NeuralNetwork_saveSynapseWeights(&net, buf);

		//assertions
		assertDoubleEqual(buf[0], net.layers[0].neurons[0].synapses[0].weight, 0.0001, t, "A1");
		assertDoubleEqual(buf[1], net.layers[0].neurons[0].synapses[1].weight, 0.0001, t, "A2");
		assertDoubleEqual(buf[2], net.layers[0].neurons[1].synapses[0].weight, 0.0001, t, "A3");
		assertDoubleEqual(buf[3], net.layers[0].neurons[1].synapses[1].weight, 0.0001, t, "A4");
		assertDoubleEqual(buf[4], net.layers[0].neurons[1].synapses[2].weight, 0.0001, t, "A5");
		assertDoubleEqual(buf[5], net.layers[0].bias.synapses[0].weight, 0.0001, t, "A6");
		assertDoubleEqual(buf[6], net.layers[0].bias.synapses[1].weight, 0.0001, t, "A7");
		assertDoubleEqual(buf[7], net.layers[0].bias.synapses[2].weight, 0.0001, t, "A8");

		assertDoubleEqual(buf[8], net.layers[1].neurons[0].synapses[0].weight, 0.0001, t, "A9");
		assertDoubleEqual(buf[9], net.layers[1].neurons[0].synapses[1].weight, 0.0001, t, "A10");
		assertDoubleEqual(buf[10], net.layers[1].neurons[1].synapses[0].weight, 0.0001, t, "A11");
		assertDoubleEqual(buf[11], net.layers[1].neurons[1].synapses[1].weight, 0.0001, t, "A12");
		assertDoubleEqual(buf[12], net.layers[1].neurons[2].synapses[0].weight, 0.0001, t, "A13");
		assertDoubleEqual(buf[13], net.layers[1].bias.synapses[0].weight, 0.0001, t, "A14");
		assertDoubleEqual(buf[14], net.layers[1].bias.synapses[1].weight, 0.0001, t, "A15");

		assertDoubleEqual(buf[15], net.layers[2].neurons[0].synapses[0].weight, 0.0001, t, "A16");
		assertDoubleEqual(buf[16], net.layers[2].neurons[0].synapses[1].weight, 0.0001, t, "A17");
		assertDoubleEqual(buf[17], net.layers[2].neurons[1].synapses[0].weight, 0.0001, t, "A18");
		assertDoubleEqual(buf[18], net.layers[2].neurons[1].synapses[1].weight, 0.0001, t, "A19");
		assertDoubleEqual(buf[19], net.layers[2].bias.synapses[0].weight, 0.0001, t, "A20");
		assertDoubleEqual(buf[20], net.layers[2].bias.synapses[1].weight, 0.0001, t, "A21");
		assertDoubleEqual(0, buf[21], 0.0001, t, "A22");

		//load custom values to Network
		buf[0] = 0.1;
		buf[1] = -0.1;
		buf[2] = 0.4;
		buf[3] = -0.3;
		buf[4] = -0.2;
		buf[5] = -0.7;
		buf[6] = 0.6;
		buf[7] = 0.6;
		buf[8] = -0.4;
		buf[9] = -0.3;
		buf[10] = -0.6;
		buf[11] = 0.7;
		buf[12] = 0.4;
		buf[13] = -0.5;
		buf[14] = -0.9;
		buf[15] = 0.3;
		buf[16] = 0.3;
		buf[17] = -0.5;
		buf[18] = -0.3;
		buf[19] = -0.2;
		buf[20] = 0.1;
		NeuralNetwork_loadSynapseWeights(&net, buf);

		assertDoubleEqual(buf[0], net.layers[0].neurons[0].synapses[0].weight, 0.001, t, "B1");
		assertDoubleEqual(buf[1], net.layers[0].neurons[0].synapses[1].weight, 0.001, t, "B2");
		assertDoubleEqual(buf[2], net.layers[0].neurons[1].synapses[0].weight, 0.001, t, "B3");
		assertDoubleEqual(buf[3], net.layers[0].neurons[1].synapses[1].weight, 0.001, t, "B4");
		assertDoubleEqual(buf[4], net.layers[0].neurons[1].synapses[2].weight, 0.001, t, "B5");
		assertDoubleEqual(buf[5], net.layers[0].bias.synapses[0].weight, 0.001, t, "B6");
		assertDoubleEqual(buf[6], net.layers[0].bias.synapses[1].weight, 0.001, t, "B7");
		assertDoubleEqual(buf[7], net.layers[0].bias.synapses[2].weight, 0.001, t, "B8");

		assertDoubleEqual(buf[8], net.layers[1].neurons[0].synapses[0].weight, 0.001, t, "B9");
		assertDoubleEqual(buf[9], net.layers[1].neurons[0].synapses[1].weight, 0.001, t, "B10");
		assertDoubleEqual(buf[10], net.layers[1].neurons[1].synapses[0].weight, 0.001, t, "B11");
		assertDoubleEqual(buf[11], net.layers[1].neurons[1].synapses[1].weight, 0.001, t, "B12");
		assertDoubleEqual(buf[12], net.layers[1].neurons[2].synapses[0].weight, 0.001, t, "B13");
		assertDoubleEqual(buf[13], net.layers[1].bias.synapses[0].weight, 0.001, t, "B14");
		assertDoubleEqual(buf[14], net.layers[1].bias.synapses[1].weight, 0.001, t, "B15");

		assertDoubleEqual(buf[15], net.layers[2].neurons[0].synapses[0].weight, 0.001, t, "B16");
		assertDoubleEqual(buf[16], net.layers[2].neurons[0].synapses[1].weight, 0.001, t, "B17");
		assertDoubleEqual(buf[17], net.layers[2].neurons[1].synapses[0].weight, 0.001, t, "B18");
		assertDoubleEqual(buf[18], net.layers[2].neurons[1].synapses[1].weight, 0.001, t, "B19");
		assertDoubleEqual(buf[19], net.layers[2].bias.synapses[0].weight, 0.001, t, "B20");
		assertDoubleEqual(buf[20], net.layers[2].bias.synapses[1].weight, 0.001, t, "B21");
		assertDoubleEqual(0, buf[21], 0.0001, t, "B22");


		//set adjustments and upload them to network
		buf[0] = 0.2;
		buf[1] = -0.2;
		buf[2] = 0.1;
		buf[3] = 0.4;
		buf[4] = 0.1;
		buf[5] = -0.3;
		buf[6] = 0.4;
		buf[7] = -0.6;
		buf[8] = 0.2;
		buf[9] = -0.1;
		buf[10] = -0.3;
		buf[11] = -0.2;
		buf[12] = 0.5;
		buf[13] = -0.5;
		buf[14] = 0.4;
		buf[15] = 0.4;
		buf[16] = -0.3;
		buf[17] = 0.2;
		buf[18] = -0.1;
		buf[19] = 0.2;
		buf[20] = 0.5;
		NeuralNetwork_adjustWeights(&net, buf);

		assertDoubleEqual(0.1+0.2, net.layers[0].neurons[0].synapses[0].weight, 0.0001, t, "C1");
		assertDoubleEqual(-0.1-0.2, net.layers[0].neurons[0].synapses[1].weight, 0.0001, t, "C2");
		assertDoubleEqual(0.4+0.1, net.layers[0].neurons[1].synapses[0].weight, 0.0001, t, "C3");
		assertDoubleEqual(-0.3+0.4, net.layers[0].neurons[1].synapses[1].weight, 0.0001, t, "C4");
		assertDoubleEqual(-0.2+0.1, net.layers[0].neurons[1].synapses[2].weight, 0.0001, t, "C5");
		assertDoubleEqual(-0.7-0.3, net.layers[0].bias.synapses[0].weight, 0.0001, t, "C6");
		assertDoubleEqual(0.6+0.4, net.layers[0].bias.synapses[1].weight, 0.0001, t, "C7");
		assertDoubleEqual(0.6-0.6, net.layers[0].bias.synapses[2].weight, 0.0001, t, "C8");

		assertDoubleEqual(-0.4+0.2, net.layers[1].neurons[0].synapses[0].weight, 0.0001, t, "C9");
		assertDoubleEqual(-0.3-0.1, net.layers[1].neurons[0].synapses[1].weight, 0.0001, t, "C10");
		assertDoubleEqual(-0.6-0.3, net.layers[1].neurons[1].synapses[0].weight, 0.0001, t, "C11");
		assertDoubleEqual(0.7-0.2, net.layers[1].neurons[1].synapses[1].weight, 0.0001, t, "C12");
		assertDoubleEqual(0.4+0.5, net.layers[1].neurons[2].synapses[0].weight, 0.0001, t, "C13");
		assertDoubleEqual(-0.5-0.5, net.layers[1].bias.synapses[0].weight, 0.0001, t, "C14");
		assertDoubleEqual(-0.9+0.4, net.layers[1].bias.synapses[1].weight, 0.0001, t, "C15");

		assertDoubleEqual(0.3+0.4, net.layers[2].neurons[0].synapses[0].weight, 0.0001, t, "C16");
		assertDoubleEqual(0.3-0.3, net.layers[2].neurons[0].synapses[1].weight, 0.0001, t, "C17");
		assertDoubleEqual(-0.5+0.2, net.layers[2].neurons[1].synapses[0].weight, 0.0001, t, "C18");
		assertDoubleEqual(-0.3-0.1, net.layers[2].neurons[1].synapses[1].weight, 0.0001, t, "C19");
		assertDoubleEqual(-0.2+0.2, net.layers[2].bias.synapses[0].weight, 0.0001, t, "C20");
		assertDoubleEqual(0.1+0.5, net.layers[2].bias.synapses[1].weight, 0.0001, t, "C21");
		assertDoubleEqual(0, buf[21], 0.0001, t, "C22");

		free(buf);
		NeuralNetwork_deinit(&net);
	}


	void testNetworkPropagations(TestCase *t) {
		NeuralNetwork net;
		createSimpleNetwork(&net);
		NeuronUnit* buf = malloc(net.synapseCount * sizeof(NeuronUnit) + 1);
		buf[net.synapseCount] = 0; //test for bound violations. this must stay 0



		NeuralNetwork_predict(&net);
		assertDoubleEqual(-0.14, net.layers[1].neurons[0].in, 0.001, t, "A1");
		assertDoubleEqual(0.4651, net.layers[1].neurons[0].out, 0.001, t, "A2");
		assertDoubleEqual(0.2, net.layers[1].neurons[1].in, 0.001, t, "A3");
		assertDoubleEqual(0.5498, net.layers[1].neurons[1].out, 0.001, t, "A4");
		assertDoubleEqual(0.86, net.layers[1].neurons[2].in, 0.001, t, "A5");
		assertDoubleEqual(0.7027, net.layers[1].neurons[2].out, 0.001, t, "A6");

		assertDoubleEqual(-0.1623, net.layers[2].neurons[0].in, 0.001, t, "A7");
		assertDoubleEqual(-0.1623, net.layers[2].neurons[0].out, 0.001, t, "A8");
		assertDoubleEqual(-0.11609, net.layers[2].neurons[1].in, 0.001, t, "A9");
		assertDoubleEqual(-0.11609, net.layers[2].neurons[1].out, 0.001, t, "A10");

		assertDoubleEqual(-0.162806, net.layers[3].neurons[0].in, 0.001, t, "A11");
		assertDoubleEqual(0.4594, net.layers[3].neurons[0].out, 0.001, t, "A12");
		assertDoubleEqual(0.689, net.layers[3].neurons[1].in, 0.001, t, "A13");
		assertDoubleEqual(0.6657, net.layers[3].neurons[1].out, 0.001, t, "A14");

		NeuronUnit errorDerivatives[] = {-0.4, 0.8};
		NeuralNetwork_saveGradient(&net, errorDerivatives, buf);

		//check deltas
		assertDoubleEqual(-0.099, net.layers[3].neurons[0].delta, 0.001, t, "B1");
		assertDoubleEqual(0.1780, net.layers[3].neurons[1].delta, 0.001, t, "B2");

		assertDoubleEqual(0.1404, net.layers[2].neurons[0].delta, 0.001, t, "B3");
		assertDoubleEqual(0.0062, net.layers[2].neurons[1].delta, 0.001, t, "B4");

		assertDoubleEqual(0.0178, net.layers[1].neurons[0].delta, 0.001, t, "B5");
		assertDoubleEqual(-0.0319, net.layers[1].neurons[1].delta, 0.001, t, "B6");
		assertDoubleEqual(0.00037, net.layers[1].neurons[2].delta, 0.001, t, "B7");


		//check gradient
		assertDoubleEqual(0.00712, buf[0], 0.0001, t, "C1");
		assertDoubleEqual(0.00014, buf[1], 0.0001, t, "C2");
		assertDoubleEqual(0.0107, buf[2], 0.0001, t, "C3");
		assertDoubleEqual(-0.01914, buf[3], 0.0001, t, "C4");
		assertDoubleEqual(0.00022, buf[4], 0.0001, t, "C5");
		assertDoubleEqual(0.0178, buf[5], 0.0001, t, "C6");
		assertDoubleEqual(-0.0319, buf[6], 0.0001, t, "C7");
		assertDoubleEqual(0.00037, buf[7], 0.0001, t, "C8");

		assertDoubleEqual(0.0653, buf[8], 0.0001, t, "C9");
		assertDoubleEqual(0.0029, buf[9], 0.0001, t, "C10");
		assertDoubleEqual(0.0772, buf[10], 0.0001, t, "C11");
		assertDoubleEqual(0.0034, buf[11], 0.0001, t, "C12");
		assertDoubleEqual(0.00435, buf[12], 0.0001, t, "C13");
		assertDoubleEqual(0.1404, buf[13], 0.0001, t, "C14");
		assertDoubleEqual(0.0062, buf[14], 0.0001, t, "C15");

		assertDoubleEqual(0.0161, buf[15], 0.0001, t, "C16");
		assertDoubleEqual(-0.0289, buf[16], 0.0001, t, "C17");
		assertDoubleEqual(0.0115, buf[17], 0.0001, t, "C18");
		assertDoubleEqual(-0.0207, buf[18], 0.0001, t, "C19");
		assertDoubleEqual(-0.0993, buf[19], 0.0001, t, "C20");
		assertDoubleEqual(0.178, buf[20], 0.0001, t, "C21");
		assertDoubleEqual(0, buf[21], 0.0001, t, "C22");


		//test add to gradient
		buf[0] = 0.8;
		buf[1] = -0.5;
		buf[2] = 0.4;
		buf[3] = -0.2;
		buf[4] = 0.4;
		buf[5] = -0.3;
		buf[6] = -0.8;
		buf[7] = -0.4;
		buf[8] = 0.3;
		buf[9] = 0.6;
		buf[10] = 0.3;
		buf[11] = -0.7;
		buf[12] = 0.8;
		buf[13] = -0.3;
		buf[14] = -0.2;
		buf[15] = 0.1;
		buf[16] = -0.6;
		buf[17] = 0.5;
		buf[18] = -0.3;
		buf[19] = 0.4;
		buf[20] = -0.2;
		NeuralNetwork_addToGradient(&net, errorDerivatives, buf);
		assertDoubleEqual(0.00712 + 0.8, buf[0], 0.0001, t, "D1");
		assertDoubleEqual(0.00014 - 0.5, buf[1], 0.0001, t, "D2");
		assertDoubleEqual(0.0107 + 0.4, buf[2], 0.0001, t, "D3");
		assertDoubleEqual(-0.01914 - 0.2, buf[3], 0.0001, t, "D4");
		assertDoubleEqual(0.00022 + 0.4, buf[4], 0.0001, t, "D5");
		assertDoubleEqual(0.0178 - 0.3, buf[5], 0.0001, t, "D6");
		assertDoubleEqual(-0.0319 - 0.8, buf[6], 0.0001, t, "D7");
		assertDoubleEqual(0.00037 - 0.4, buf[7], 0.0001, t, "D8");

		assertDoubleEqual(0.0653 + 0.3, buf[8], 0.0001, t, "D9");
		assertDoubleEqual(0.0029 + 0.6, buf[9], 0.0001, t, "D10");
		assertDoubleEqual(0.0772 + 0.3, buf[10], 0.0001, t, "D11");
		assertDoubleEqual(0.0034 - 0.7, buf[11], 0.0001, t, "D12");
		assertDoubleEqual(0.00435 + 0.8, buf[12], 0.0001, t, "D13");
		assertDoubleEqual(0.1404 - 0.3, buf[13], 0.0001, t, "D14");
		assertDoubleEqual(0.0062 - 0.2, buf[14], 0.0001, t, "D15");

		assertDoubleEqual(0.0161 + 0.1, buf[15], 0.0001, t, "D16");
		assertDoubleEqual(-0.0289 - 0.6, buf[16], 0.0001, t, "D17");
		assertDoubleEqual(0.0115 + 0.5, buf[17], 0.0001, t, "D18");
		assertDoubleEqual(-0.0207 - 0.3, buf[18], 0.0001, t, "D19");
		assertDoubleEqual(-0.0993 + 0.4, buf[19], 0.0001, t, "D20");
		assertDoubleEqual(0.178- 0.2, buf[20], 0.0001, t, "D21");
		assertDoubleEqual(0, buf[21], 0.0001, t, "D22");

		free(buf);
		NeuralNetwork_deinit(&net);
	}


//LAYER FUNCTIONS
	void testLayerInitializations(TestCase *t) {
		//Test output layer
		NetworkLayer layer;
		NetworkLayerStructure str = {
			.connectionType = NetworkLayer_FULLY_CONNECTED,
			.activatorType = NeuronActivator_SIGMOID,
			.neuronCount = 4
		};

		NetworkLayer_initOutput(&layer, &str);
		assertIntEqual(str.neuronCount, layer.neuronCount, t, "A1");
		assertIntEqual(0, layer.synapseCount, t, "A2");
		assertPtrEqual(NeuronActivator_sigmoid, layer.activator.inToOut, t, "A3");
		assertPtrEqual(NeuronActivator_sigmoidDerivative, layer.activator.inToDerivative, t, "A4");
		assertIntEqual(0, layer.bias.synapseCount, t, "A5");
		for (int i=0; i<layer.neuronCount; ++i ) assertIntEqual(0, layer.neurons[i].synapseCount, t, "A6");
		NetworkLayer_deinit(&layer);


		//Test fully connected layer
		str.connectionType = NetworkLayer_FULLY_CONNECTED;
		NetworkLayer_initFullyConnected(&layer, &str, 3);
		assertIntEqual(str.neuronCount, layer.neuronCount, t, "B1");
		assertIntEqual(4*3 + 3, layer.synapseCount, t, "B2");
		assertPtrEqual(NeuronActivator_sigmoid, layer.activator.inToOut, t, "B3");
		assertPtrEqual(NeuronActivator_sigmoidDerivative, layer.activator.inToDerivative, t, "B4");

		//check bias
		assertIntEqual(3, layer.bias.synapseCount, t, "B5");
		for (int i=0; i<layer.bias.synapseCount; ++i) assertIntEqual(i, layer.bias.synapses[i].targetIndex, t, "B6");

		//check neurons one by one
		for (int i=0; i<layer.neuronCount; ++i) {
			assertIntEqual(3, layer.neurons[i].synapseCount, t, "B7");
			for (int j=0; j<layer.bias.synapseCount; ++j) assertIntEqual(j, layer.neurons[i].synapses[j].targetIndex, t, "B8");
		}
		NetworkLayer_deinit(&layer);
	}


	void testLayerWeightsManagement(TestCase *t) {
		NeuralNetwork net;
		createSimpleNetwork(&net);
		NetworkLayer *layer = &net.layers[0];
		NeuronUnit* buf = malloc(net.layers[0].synapseCount * sizeof(NeuronUnit));

		//randomize synapse weights and save them
		srand(time(NULL));
		NetworkLayer_randomSynapses(layer);
		NetworkLayer_saveSynapseWeights(layer, buf);

		assertDoubleEqual(layer->neurons[0].synapses[0].weight, buf[0], 0.001, t, "A1");
		assertDoubleEqual(layer->neurons[0].synapses[1].weight, buf[1], 0.001, t, "A2");
		assertDoubleEqual(layer->neurons[1].synapses[0].weight, buf[2], 0.001, t, "A3");
		assertDoubleEqual(layer->neurons[1].synapses[1].weight, buf[3], 0.001, t, "A4");
		assertDoubleEqual(layer->neurons[1].synapses[2].weight, buf[4], 0.001, t, "A5");
		assertDoubleEqual(layer->bias.synapses[0].weight, buf[5], 0.001, t, "A6");
		assertDoubleEqual(layer->bias.synapses[1].weight, buf[6], 0.001, t, "A7");
		assertDoubleEqual(layer->bias.synapses[2].weight, buf[7], 0.001, t, "A8");

		//load custom values to Layer
		buf[0] = 0.1;
		buf[1] = 0.2;
		buf[2] = 0.3;
		buf[3] = 0.4;
		buf[4] = 0.5;
		buf[5] = 0.6;
		buf[6] = 0.7;
		buf[7] = 0.8;
		NetworkLayer_loadSynapseWeights(layer, buf);

		assertDoubleEqual(buf[0], layer->neurons[0].synapses[0].weight, 0.001, t, "B1");
		assertDoubleEqual(buf[1], layer->neurons[0].synapses[1].weight, 0.001, t, "B2");
		assertDoubleEqual(buf[2], layer->neurons[1].synapses[0].weight, 0.001, t, "B3");
		assertDoubleEqual(buf[3], layer->neurons[1].synapses[1].weight, 0.001, t, "B4");
		assertDoubleEqual(buf[4], layer->neurons[1].synapses[2].weight, 0.001, t, "B5");
		assertDoubleEqual(buf[5], layer->bias.synapses[0].weight, 0.001, t, "B6");
		assertDoubleEqual(buf[6], layer->bias.synapses[1].weight, 0.001, t, "B7");
		assertDoubleEqual(buf[7], layer->bias.synapses[2].weight, 0.001, t, "B8");


		//set adjustments and upload them to layer
		buf[0] = 0.2;
		buf[1] = 0.1;
		buf[2] = 0.3;
		buf[3] = 0.1;
		buf[4] = 0.2;
		buf[5] = 0.2;
		buf[6] = 0.8;
		buf[7] = -0.7;
		NetworkLayer_adjustWeights(layer, buf);

		assertDoubleEqual(0.3, layer->neurons[0].synapses[0].weight, 0.001, t, "C1");
		assertDoubleEqual(0.3, layer->neurons[0].synapses[1].weight, 0.001, t, "C2");
		assertDoubleEqual(0.6, layer->neurons[1].synapses[0].weight, 0.001, t, "C3");
		assertDoubleEqual(0.5, layer->neurons[1].synapses[1].weight, 0.001, t, "C4");
		assertDoubleEqual(0.7, layer->neurons[1].synapses[2].weight, 0.001, t, "C5");
		assertDoubleEqual(0.8, layer->bias.synapses[0].weight, 0.001, t, "C6");
		assertDoubleEqual(1.5, layer->bias.synapses[1].weight, 0.001, t, "C7");
		assertDoubleEqual(0.1, layer->bias.synapses[2].weight, 0.001, t, "C8");

		free(buf);
		NeuralNetwork_deinit(&net);
	}

	void testLayerFiring(TestCase *t) {
		NeuralNetwork net;
		createSimpleNetwork(&net);

		//fire layer 0
		NetworkLayer_reset(&(net.layers[1]));
		NetworkLayer_fire(&(net.layers[0]));
		assertDoubleEqual(0.1 * 0.4 + 0.2*0.6 + 1 * (-0.3), net.layers[1].neurons[0].in, 0.0001, t, "A1"); //-0.14
		assertDoubleEqual(0.5*0.6 - 0.1*1, net.layers[1].neurons[1].in, 0.0001, t, "A2"); //0.2
		assertDoubleEqual(-0.25 * 0.4 + 0.6*0.1 + 1*0.9, net.layers[1].neurons[2].in, 0.0001, t, "A3"); //0.86

		//activate layer 1
		NetworkLayer_activate(&(net.layers[1]));
		assertDoubleEqual(NeuronActivator_sigmoid(-0.14), net.layers[1].neurons[0].out, 0.0001, t, "B1"); //-0.14
		assertDoubleEqual(NeuronActivator_sigmoid(0.2), net.layers[1].neurons[1].out, 0.0001, t, "B2"); //0.2
		assertDoubleEqual(NeuronActivator_sigmoid(0.86), net.layers[1].neurons[2].out, 0.0001, t, "B3"); //0.86
		assertDoubleEqual(1, net.layers[1].bias.out, 0.0001, t, "B4");

		//check that in are not affected
		assertDoubleEqual(0.1 * 0.4 + 0.2*0.6 + 1 * (-0.3), net.layers[1].neurons[0].in, 0.0001, t, "C1"); //-0.14
		assertDoubleEqual(0.5*0.6 - 0.1*1, net.layers[1].neurons[1].in, 0.0001, t, "C2"); //0.2
		assertDoubleEqual(-0.25 * 0.4 + 0.6*0.1 + 1*0.9, net.layers[1].neurons[2].in, 0.0001, t, "C3"); //0.86

		//reset all inputs
		NetworkLayer_reset(&(net.layers[1]));
		assertDoubleEqual(0, net.layers[1].neurons[0].in, 0.0001, t, "D1");
		assertDoubleEqual(0, net.layers[1].neurons[1].in, 0.0001, t, "D2");
		assertDoubleEqual(0, net.layers[1].neurons[2].in, 0.0001, t, "D3");
		assertDoubleEqual(1, net.layers[1].bias.out, 0.0001, t, "D4"); //output of bias must always be 1

		NeuralNetwork_deinit(&net);
	}

	void testLayerDeltaAndGradient(TestCase *t) {
		NeuralNetwork net;
		createSimpleNetwork(&net);

		//fire the whole network
		NetworkLayer_reset(&(net.layers[1]));
		NetworkLayer_reset(&(net.layers[2]));
		NetworkLayer_reset(&(net.layers[3]));
		NetworkLayer_fire(&(net.layers[0]));
		NetworkLayer_activate(&(net.layers[1]));
		NetworkLayer_fire(&(net.layers[1]));
		NetworkLayer_activate(&(net.layers[2]));
		NetworkLayer_fire(&(net.layers[2]));
		NetworkLayer_activate(&(net.layers[3]));

		//calculate deltas and gradients
		NeuronUnit errorDerivative[] = {-0.4, 0.8};
		NeuronUnit grads[9]; //we need actually 8, the 9th is just to be sure that the function is is falsely not crossing the bounds

		//LAYER 3
		net.layers[3].bias.delta = 0; //this should stay that way
		grads[6] = 0; //it must stay like that

		NetworkLayer_calculateDeltaFromErrorDerivatives(&(net.layers[3]), errorDerivative);
		assertDoubleEqual(-0.0993, net.layers[3].neurons[0].delta, 0.0001, t, "A1");
		assertDoubleEqual(0.1780, net.layers[3].neurons[1].delta, 0.0001, t, "A2");
		assertDoubleEqual(0, net.layers[3].bias.delta, 0.0001, t, "A3");

		NetworkLayer_saveGradient(&(net.layers[2]), grads);
		assertDoubleEqual(0.1404, net.layers[2].neurons[0].delta, 0.0001, t, "B1");
		assertDoubleEqual(0.0062, net.layers[2].neurons[1].delta, 0.0001, t, "B2");
		assertDoubleEqual(0, net.layers[2].bias.delta, 0.0001, t, "B3");
		assertDoubleEqual(0.0161, grads[0], 0.0001, t, "B4");
		assertDoubleEqual(-0.0289, grads[1], 0.0001, t, "B5");
		assertDoubleEqual(0.0115, grads[2], 0.0001, t, "B6");
		assertDoubleEqual(-0.0207, grads[3], 0.0001, t, "B7");
		assertDoubleEqual(-0.0993, grads[4], 0.0001, t, "B8");
		assertDoubleEqual(0.178, grads[5], 0.0001, t, "B9");
		assertDoubleEqual(0, grads[6], 0.0001, t, "B10");

		//add to gradient
		grads[0] = 0.7;
		grads[1] = -0.2;
		grads[2] = 0.4;
		grads[3] = -0.5;
		grads[4] = -0.4;
		grads[5] = 0.1;
		NetworkLayer_addToGradient(&(net.layers[2]), grads);
		assertDoubleEqual(0.1404, net.layers[2].neurons[0].delta, 0.0001, t, "B11");
		assertDoubleEqual(0.0062, net.layers[2].neurons[1].delta, 0.0001, t, "B12");
		assertDoubleEqual(0, net.layers[2].bias.delta, 0.0001, t, "B13");
		assertDoubleEqual(0.0161 + 0.7, grads[0], 0.0001, t, "B14");
		assertDoubleEqual(-0.0289 - 0.2, grads[1], 0.0001, t, "B15");
		assertDoubleEqual(0.0115 + 0.4, grads[2], 0.0001, t, "B16");
		assertDoubleEqual(-0.0207 - 0.5, grads[3], 0.0001, t, "B17");
		assertDoubleEqual(-0.0993 - 0.4, grads[4], 0.0001, t, "B18");
		assertDoubleEqual(0.178 + 0.1, grads[5], 0.0001, t, "B19");
		assertDoubleEqual(0, grads[6], 0.0001, t, "B20");


		//LAYER 2
		net.layers[2].bias.delta = 0; //this should stay that way
		grads[7] = 0; //it must stay like that
		NetworkLayer_saveGradient(&(net.layers[1]), grads);
		assertDoubleEqual(0.0653, grads[0], 0.0001, t, "C1");
		assertDoubleEqual(0.0029, grads[1], 0.0001, t, "C2");
		assertDoubleEqual(0.0772, grads[2], 0.0001, t, "C3");
		assertDoubleEqual(0.0034, grads[3], 0.0001, t, "C4");
		assertDoubleEqual(0.00435, grads[4], 0.0001, t, "C5");
		assertDoubleEqual(0.1404, grads[5], 0.0001, t, "C6");
		assertDoubleEqual(0.0062, grads[6], 0.0001, t, "C7");
		assertDoubleEqual(0, grads[7], 0.0001, t, "C8");


		//add to gradient
		grads[0] = 0.7;
		grads[1] = -0.2;
		grads[2] = 0.4;
		grads[3] = -0.5;
		grads[4] = -0.4;
		grads[5] = 0.1;
		grads[6] = 0.4;
		NetworkLayer_addToGradient(&(net.layers[1]), grads);
		assertDoubleEqual(0.0178, net.layers[1].neurons[0].delta, 0.0001, t, "C9");
		assertDoubleEqual(-0.0319, net.layers[1].neurons[1].delta, 0.0001, t, "C10");
		assertDoubleEqual(0.00037, net.layers[1].neurons[2].delta, 0.0001, t, "C11");
		assertDoubleEqual(0, net.layers[1].bias.delta, 0.0001, t, "C12");
		assertDoubleEqual(0.0653 + 0.7, grads[0], 0.0001, t, "C13");
		assertDoubleEqual(0.0029 - 0.2, grads[1], 0.0001, t, "C14");
		assertDoubleEqual(0.0772 + 0.4, grads[2], 0.0001, t, "C15");
		assertDoubleEqual(0.0034 - 0.5, grads[3], 0.0001, t, "C16");
		assertDoubleEqual(0.00435 - 0.4, grads[4], 0.0001, t, "C17");
		assertDoubleEqual(0.1404 + 0.1, grads[5], 0.0001, t, "C18");
		assertDoubleEqual(0.0062 + 0.4, grads[6], 0.0001, t, "C19");
		assertDoubleEqual(0, grads[7], 0.0001, t, "C20");



		//LAYER 1
		net.layers[1].bias.delta = 0; //this should stay that way
		grads[8] = 0; //it must stay like that
		NetworkLayer_saveGradient(&(net.layers[0]), grads);
		assertDoubleEqual(0.00712, grads[0], 0.0001, t, "D1");
		assertDoubleEqual(0.00014, grads[1], 0.0001, t, "D2");
		assertDoubleEqual(0.0107, grads[2], 0.0001, t, "D3");
		assertDoubleEqual(-0.01914, grads[3], 0.0001, t, "D4");
		assertDoubleEqual(0.00022, grads[4], 0.0001, t, "D5");
		assertDoubleEqual(0.0178, grads[5], 0.0001, t, "D6");
		assertDoubleEqual(-0.0319, grads[6], 0.0001, t, "D7");
		assertDoubleEqual(0.00037, grads[7], 0.0001, t, "D8");
		assertDoubleEqual(0, grads[8], 0.0001, t, "D9");


		//addToGradient
		//add to gradient
		grads[0] = 0.7;
		grads[1] = -0.2;
		grads[2] = 0.4;
		grads[3] = -0.5;
		grads[4] = -0.4;
		grads[5] = 0.1;
		grads[6] = 0.4;
		grads[7] = -0.7;
		NetworkLayer_addToGradient(&(net.layers[0]), grads);
		assertDoubleEqual(0.00712 + 0.7, grads[0], 0.0001, t, "D14");
		assertDoubleEqual(0.00014 - 0.2, grads[1], 0.0001, t, "D15");
		assertDoubleEqual(0.0107 + 0.4, grads[2], 0.0001, t, "D16");
		assertDoubleEqual(-0.01914 - 0.5, grads[3], 0.0001, t, "D17");
		assertDoubleEqual(0.00022 - 0.4, grads[4], 0.0001, t, "D18");
		assertDoubleEqual(0.0178 + 0.1, grads[5], 0.0001, t, "D19");
		assertDoubleEqual(-0.0319 + 0.4, grads[6], 0.0001, t, "D20");
		assertDoubleEqual(0.00037 - 0.7, grads[7], 0.0001, t, "D21");
		assertDoubleEqual(0, grads[8], 0.0001, t, "D22");


		NeuralNetwork_deinit(&net);
	}



//Neuron tests
	void testNeuronWeightsManagement(TestCase *t) { //save, load and adjust
		NetworkLayer layer1;
		NetworkLayer layer2;
		createSimpleStructure(&layer1, &layer2);

		Neuron *test = &(layer1.neurons[1]);
		NeuronUnit* buf = malloc(test->synapseCount * sizeof(NeuronUnit));

		//randomize and save weights
		srand(time(NULL));
		Neuron_randomSynapses(test);
		Neuron_saveSynapseWeights(test, buf);

		for (unsigned short int i = 0; i< test->synapseCount; ++i) assertDoubleEqual(buf[i], test->synapses[i].weight, 0.001, t, "A1");

		//now give other values to buf, and load it to neuron
		buf[0] = 0.3;
		buf[1] = -0.7;
		buf[2] = 0.33;
		Neuron_loadSynapseWeights(test, buf);

		//test result
		assertDoubleEqual(0.3, test->synapses[0].weight, 0.001, t, "A2");
		assertDoubleEqual(-0.7, test->synapses[1].weight, 0.001, t, "A3");
		assertDoubleEqual(0.33, test->synapses[2].weight, 0.001, t, "A4");

		//adjust the weights by specific values
		buf[0] = 0.1;
		buf[1] = 0.2;
		buf[2] = 0.5;
		Neuron_adjustWeights(test, buf);

		//test results
		assertDoubleEqual( 0.4, test->synapses[0].weight, 0.001, t, "B1");
		assertDoubleEqual(-0.5, test->synapses[1].weight, 0.001, t, "B2");
		assertDoubleEqual(0.83, test->synapses[2].weight, 0.001, t, "B3");

		free(buf);
		NetworkLayer_deinit(&layer1);
		NetworkLayer_deinit(&layer2);
	}

	void testNeuronFiring(TestCase *t) {
		NetworkLayer layer1;
		NetworkLayer layer2;
		createSimpleStructure(&layer1, &layer2);

		//reset all neurons of layer 2
		for (unsigned short int i = 0; i<layer2.neuronCount; ++i) {
			Neuron_reset(&(layer2.neurons[i]));
			assertDoubleEqual(0, layer2.neurons[i].in, 0.001, t, "C1");
		}
		Neuron_reset(&(layer2.bias));
		assertDoubleEqual(0, layer2.bias.in, 0.001, t, "C2");

		//fire neuron 0
		Neuron_fire(&(layer1.neurons[0]));
		assertDoubleEqual(0.04, layer2.neurons[0].in, 0.001, t, "D1");
		assertDoubleEqual(0, layer2.neurons[1].in, 0.001, t, "D2");
		assertDoubleEqual(-0.1, layer2.neurons[2].in, 0.001, t, "D3");
		assertDoubleEqual(0, layer2.bias.in, 0.001, t, "D4");

		//fire neuron 1
		Neuron_fire(&(layer1.neurons[1]));
		assertDoubleEqual(0.04 + 0.12, layer2.neurons[0].in, 0.001, t, "E1");
		assertDoubleEqual(0.3, layer2.neurons[1].in, 0.001, t, "E2");
		assertDoubleEqual(-0.1 + 0.06, layer2.neurons[2].in, 0.001, t, "E3");
		assertDoubleEqual(0, layer2.bias.in, 0.001, t, "E4");

		//fire bias neuron
		Neuron_fire(&(layer1.bias));
		assertDoubleEqual(0.04 + 0.12 -0.3, layer2.neurons[0].in, 0.001, t, "F1");
		assertDoubleEqual(0.3 -0.1, layer2.neurons[1].in, 0.001, t, "F2");
		assertDoubleEqual(-0.1 + 0.06 + 0.9, layer2.neurons[2].in, 0.001, t, "F3");
		assertDoubleEqual(0, layer2.bias.in, 0.001, t, "F4");

		//test activations
		Neuron_activate(&(layer2.neurons[0]), &dummyActivation);
		Neuron_activate(&(layer2.neurons[1]), &dummyActivation);
		Neuron_activate(&(layer2.neurons[2]), &dummyActivation);
		assertDoubleEqual(-0.1351, layer2.neurons[0].out, 0.00001, t, "G1");
		assertDoubleEqual(0.21, layer2.neurons[1].out, 0.00001, t, "G2");
		assertDoubleEqual(1.0449, layer2.neurons[2].out, 0.00001, t, "G3");

		NetworkLayer_deinit(&layer1);
		NetworkLayer_deinit(&layer2);
	}

	void testNeuronDeltaAndGradient(TestCase *t) {
		NetworkLayer layer1;
		NetworkLayer layer2;
		createSimpleStructure(&layer1, &layer2);

		for (unsigned short int i = 0; i<layer2.neuronCount; ++i) {
			Neuron_reset(&(layer2.neurons[i]));
		}

		Neuron_fire(&(layer1.neurons[0]));
		Neuron_fire(&(layer1.neurons[1]));
		Neuron_fire(&(layer1.bias));
		Neuron_activate(&(layer2.neurons[0]), &dummyActivation);
		Neuron_activate(&(layer2.neurons[1]), &dummyActivation);
		Neuron_activate(&(layer2.neurons[2]), &dummyActivation);

		//calculate delta from errors
		Neuron_calculateDeltaFromErrorDerivative(&(layer2.neurons[0]), *dummyActivationDerivative, 4);
		Neuron_calculateDeltaFromErrorDerivative(&(layer2.neurons[1]), *dummyActivationDerivative, 1.5);
		Neuron_calculateDeltaFromErrorDerivative(&(layer2.neurons[2]), *dummyActivationDerivative, -0.5);
		assertDoubleEqual(3.72, layer2.neurons[0].delta, 0.00001, t, "A1");
		assertDoubleEqual(1.65, layer2.neurons[1].delta, 0.00001, t, "A2");
		assertDoubleEqual(-0.715, layer2.neurons[2].delta, 0.00001, t, "A3");

		//calculate  first layers delta
		//NOTE; layer1 neurons have no input, only output. And it does not have a sense to calculate the delta value for them.
		//For the purose of this tests, we will assume that layer1 was a hidden layer, and we will give it some dummy input values
		//so we can calculate its delta
		layer1.neurons[0].in = 0.2;
		layer1.neurons[1].in = 0.4;
		NeuronUnit buf[3];
		Neuron_saveGradient(&(layer1.neurons[0]), *dummyActivationDerivative, buf);
		assertDoubleEqual( (layer1.neurons[0].in * 0.5 + 1) * (3.72*0.1 + (-0.715)*(-0.25)), layer1.neurons[0].delta, 0.00001, t, "B1");
		assertDoubleEqual(layer1.neurons[0].out * layer2.neurons[0].delta, buf[0], 0.00001, t, "B2");
		assertDoubleEqual(layer1.neurons[0].out * layer2.neurons[2].delta, buf[1], 0.00001, t, "B3");

		//add to gradient
		buf[0] = -0.1;
		buf[1] = 0.3;
		Neuron_addToGradient(&(layer1.neurons[0]), *dummyActivationDerivative, buf);
		assertDoubleEqual(layer1.neurons[0].out * layer2.neurons[0].delta - 0.1, buf[0], 0.00001, t, "B4");
		assertDoubleEqual(layer1.neurons[0].out * layer2.neurons[2].delta + 0.3, buf[1], 0.00001, t, "B5");



		Neuron_saveGradient(&(layer1.neurons[1]), *dummyActivationDerivative, buf);
		assertDoubleEqual( (layer1.neurons[1].in * 0.5 + 1) * (3.72*0.2 + 1.65*0.5 + (-0.715)*0.1), layer1.neurons[1].delta, 0.00001, t, "C1");
		assertDoubleEqual(layer1.neurons[1].out * layer2.neurons[0].delta, buf[0], 0.00001, t, "C2");
		assertDoubleEqual(layer1.neurons[1].out * layer2.neurons[1].delta, buf[1], 0.00001, t, "C3");
		assertDoubleEqual(layer1.neurons[1].out * layer2.neurons[2].delta, buf[2], 0.00001, t, "C4");

		//add to gradient
		buf[0] = -0.1;
		buf[1] = 0.2;
		buf[2] = 0.5;
		Neuron_addToGradient(&(layer1.neurons[1]), *dummyActivationDerivative, buf);
		assertDoubleEqual(layer1.neurons[1].out * layer2.neurons[0].delta - 0.1, buf[0], 0.00001, t, "C5");
		assertDoubleEqual(layer1.neurons[1].out * layer2.neurons[1].delta + 0.2, buf[1], 0.00001, t, "C6");
		assertDoubleEqual(layer1.neurons[1].out * layer2.neurons[2].delta + 0.5, buf[2], 0.00001, t, "C7");



		Neuron_saveGradient(&(layer1.bias), NULL, buf);
		assertDoubleEqual(1 * layer2.neurons[0].delta, buf[0], 0.00001, t, "D1");
		assertDoubleEqual(1 * layer2.neurons[1].delta, buf[1], 0.00001, t, "D2");
		assertDoubleEqual(1 * layer2.neurons[2].delta, buf[2], 0.00001, t, "D3");

		//add to gradient
		buf[0] = 0.3;
		buf[1] = 0.6;
		buf[2] = 0.2;
		Neuron_addToGradient(&(layer1.bias), NULL, buf);
		assertDoubleEqual(1 * layer2.neurons[0].delta + 0.3, buf[0], 0.00001, t, "D4");
		assertDoubleEqual(1 * layer2.neurons[1].delta + 0.6, buf[1], 0.00001, t, "D5");
		assertDoubleEqual(1 * layer2.neurons[2].delta + 0.2, buf[2], 0.00001, t, "D6");
	}



int main() {
	TestCase t;

//NEURON
	t.name = "testNeuronWeightsManagement";
	testNeuronWeightsManagement(&t);

	t.name = "testNeuronFiring";
	testNeuronFiring(&t);

	t.name = "testNeuronDeltaAndGradient";
	testNeuronDeltaAndGradient(&t);


//LAYER
	t.name = "testLayerInitializations";
	testLayerInitializations(&t);

	t.name = "testLayerWeightsManagement";
	testLayerWeightsManagement(&t);

	t.name = "testLayerFiring";
	testLayerFiring(&t);

	t.name = "testLayerDeltaAndGradient";
	testLayerDeltaAndGradient(&t);


//NETWORK
	t.name = "testNetworkInit";
	testNetworkInit(&t);

	t.name = "testNetworkWeightsManagement";
	testNetworkWeightsManagement(&t);

	t.name = "testNetworkPropagations";
	testNetworkPropagations(&t);
}
