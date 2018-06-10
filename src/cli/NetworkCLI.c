#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "NetworkCLI.h"

	typedef struct {
		char** tokens;
		int length;
	} Command;




	void NetworkCLI_readCommand(char* buf, Command *com, int len) {
		fgets(buf, len, stdin);

		//remove final \n
		char *pos;
		if ((pos=strchr(buf, '\n')) != NULL) *pos = '\0';


		char *next = strtok(buf, " ");
		com->length = 0;
		while (next!=NULL) {
			int len = strlen(next);
			if (len>0) {
				com->tokens[ com->length ] = next;
				com->length++;
			}
			next = strtok(NULL, " ");
		}
	}



//commands
	void NetworkCLI_reportWeights(NeuralNetwork *net) {
		unsigned long int count = net->synapseCount;
		NeuronUnit *buf = malloc(count * sizeof(NeuronUnit));

		NeuralNetwork_saveSynapseWeights(net, buf);
		for (int i=0; i<count; i++) printf("%1.2f, ", buf[i]);

		free(buf);
		printf("\n");
	}

	void NetworkCLI_reportMinWeights(Command *com, BPTrainer *online, BPTrainer *stochastic) {
		if (com->length <= 1) {
			printf("Please specify target trainer\n");
			return;
		}

		//choose trainter
		BPTrainer *target;
		if (strcmp(com->tokens[1], "online") == 0) target = online;
		else if (strcmp(com->tokens[1], "stochastic") == 0) target = stochastic;
		else {
			printf("Unknown target trainer: %s\n", com->tokens[1]);
			return;

		}

		unsigned long int count = target->network->synapseCount;
		NeuronUnit *minWeights = target->minimum.weights;
		for (int i=0; i<count; i++) printf("%1.2f, ", minWeights[i]);
		printf("\n");
	}

	void NetworkCLI_loadMinWeights(Command *com, NeuralNetwork *net, BPTrainer *online, BPTrainer *stochastic) {
		if (com->length <= 1) {
			printf("Please specify target trainer\n");
			return;
		}

		//choose trainter
		BPTrainer *target;
		if (strcmp(com->tokens[1], "online") == 0) target = online;
		else if (strcmp(com->tokens[1], "stochastic") == 0) target = stochastic;
		else {
			printf("Unknown target trainer: %s\n", com->tokens[1]);
			return;

		}

		NeuralNetwork_loadSynapseWeights(net, target->minimum.weights);
	}

	void NetworkCLI_predict(NeuralNetwork *net, Command *com) {
		int inputLen = net->layers[0].neuronCount;
		if (com->length - 1 != inputLen) {
			printf("Input size must be %d, but %d was found\n", inputLen, com->length - 1);
			return;
		}

		//initialize input buffer
		NeuronUnit buf[inputLen];
		char* ch;

		for (int i=1; i<com->length; ++i) {
			buf[i-1] = (NeuronUnit) strtod(com->tokens[i], &ch);
			if (*ch != '\0') {
				printf("Not a number: %s\n", com->tokens[i]);
				return;
			}
		}

		//feed it to network
		NetworkLayer *inpLayer = net->layers;
		for (int i=0; i<inputLen; i++) {
			inpLayer->neurons[i].out = buf[i];
		}

		NeuralNetwork_predict(net);
		NetworkLayer *outLayer = net->layers + net->layerCount - 1;
		int outLength = outLayer->neuronCount;
		for (int i=0; i<outLength; i++) printf("%1.2f ", outLayer->neurons[i].out);
		printf("\n");
	}

	void NetworkCLI_trainOnline(BPTrainer *trainer, Command *com) {
		char* check;
		NeuronUnit learningRate = 0.1;
		NeuronUnit momentum = 0;
		unsigned int times = 10000;
		char debug = 0;

		//read times
		if (com->length > 1) {
			times = strtol(com->tokens[1], &check, 10);
			if (*check != '\0') {
				printf("Not an integer: %s\n", com->tokens[1]);
				return;
			}
		}

		//read learning rate
		if (com->length > 2) {
			learningRate = strtod(com->tokens[2], &check);
			if (*check != '\0') {
				printf("Not a number: %s\n", com->tokens[2]);
				return;
			}
		}

		//read momentum
		if (com->length > 3) {
			momentum = strtod(com->tokens[3], &check);
			if (*check != '\0') {
				printf("Not an integer: %s\n", com->tokens[3]);
				return;
			}
		}

		//read debug
		if (com->length > 4) {
			debug = strtol(com->tokens[4], &check, 10);
			if (*check != '\0') {
				printf("Not an integer: %s\n", com->tokens[4]);
				return;
			}
		}

		//save current weights to compare later
		NeuralNetwork *net = trainer->network;
		NeuronUnit *startPoint = malloc(net->synapseCount * sizeof(NeuronUnit));
		NeuralNetwork_saveSynapseWeights(net, startPoint);

		//train
		TrainDataProvider_reset(trainer->provider, times);
		BPTrainer_trainOnline(trainer, learningRate, momentum, debug);

		//report the difference
		NeuronUnit *endPoint = malloc(net->synapseCount * sizeof(NeuronUnit));
		NeuralNetwork_saveSynapseWeights(net, endPoint);
		printf("Weights moved by:\n");
		for (unsigned long int i=0, len = net->synapseCount; i<len; ++i) {
			printf("%1.2f ", endPoint[i] - startPoint[i]);
		}
		printf("\n\n");
		free(startPoint);
		free(endPoint);
	}

	void NetworkCLI_trainStochastic(BPTrainer *trainer, Command *com) {
		char* check;
		NeuronUnit learningRate = 0.1;
		NeuronUnit momentum = 0;
		unsigned int times = 10000;
		unsigned int updateEvery = 25;
		char debug = 0;

		//read times
		if (com->length > 1) {
			times = strtol(com->tokens[1], &check, 10);
			if (*check != '\0') {
				printf("Not an integer: %s\n", com->tokens[1]);
				return;
			}
		}

		if (com->length > 2) {
			updateEvery = strtol(com->tokens[2], &check, 10);
			if (*check != '\0') {
				printf("Not an integer: %s\n", com->tokens[2]);
				return;
			}
		}

		//read learning rate
		if (com->length > 3) {
			learningRate = strtod(com->tokens[3], &check);
			if (*check != '\0') {
				printf("Not a number: %s\n", com->tokens[3]);
				return;
			}
		}

		//read momentum
		if (com->length > 4) {
			momentum = strtod(com->tokens[4], &check);
			if (*check != '\0') {
				printf("Not an integer: %s\n", com->tokens[4]);
				return;
			}
		}

		//read debug
		if (com->length > 5) {
			debug = strtol(com->tokens[5], &check, 10);
			if (*check != '\0') {
				printf("Not an integer: %s\n", com->tokens[5]);
				return;
			}
		}

		//save current weights to compare later
		NeuralNetwork *net = trainer->network;
		NeuronUnit *startPoint = malloc(net->synapseCount * sizeof(NeuronUnit));
		NeuralNetwork_saveSynapseWeights(net, startPoint);

		//train
		TrainDataProvider_reset(trainer->provider, times);
		BPTrainer_trainStochastic(trainer, updateEvery, learningRate, momentum, debug);

		//report the difference
		NeuronUnit *endPoint = malloc(net->synapseCount * sizeof(NeuronUnit));
		NeuralNetwork_saveSynapseWeights(net, endPoint);
		printf("Weights moved by:\n");
		for (unsigned long int i=0, len = net->synapseCount; i<len; ++i) {
			printf("%1.2f ", endPoint[i] - startPoint[i]);
		}
		printf("\n\n");
		free(startPoint);
		free(endPoint);
	}

	void NetworkCLI_setWeights(NeuralNetwork *net, Command *com) {
		if (com->length - 1 != net->synapseCount) {
			printf("Weight length must be %ld, but %d was found\n", net->synapseCount, com->length - 1);
			return;
		}

		NeuronUnit buf[net->synapseCount];
		char *ch;
		for (int i=1; i<=net->synapseCount; ++i) {
			buf[i-1] = (NeuronUnit) strtod(com->tokens[i], &ch);
			if (*ch != '\0') {
				printf("Not a number: %s\n", com->tokens[i]);
				return;
			}
		}

		NeuralNetwork_loadSynapseWeights(net, buf);
	}


	void NetworkCLI_randomWeights(NeuralNetwork *net) {
		NeuralNetwork_randomSynapses(net);
	}



	void NetworkCLI_start(
		NeuralNetwork *net,
		TrainDataProvider *provider,
		void (*onlineBPErrorFunction)(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients),
		void (*stochasticBPErrorFunction)(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients)) {
		char buf[2048];
		Command com;
		com.tokens = malloc(100 * sizeof(char*));

		BPTrainer onlineBP;
		BPTrainer stochasticBP;
		BPTrainer_init(&onlineBP, net, provider, *onlineBPErrorFunction);
		BPTrainer_init(&stochasticBP, net, provider, *stochasticBPErrorFunction);


		//loop
		while(1) {
			printf(">");
			NetworkCLI_readCommand(buf, &com, 2048);

			if (com.length == 0) continue;

			if (strcmp(com.tokens[0], "exit") == 0) break;
			else if (strcmp(com.tokens[0], "weights") == 0) NetworkCLI_reportWeights(net);
			else if (strcmp(com.tokens[0], "predict") == 0) NetworkCLI_predict(net, &com);
			else if (strcmp(com.tokens[0], "online") == 0) NetworkCLI_trainOnline(&onlineBP, &com);
			else if (strcmp(com.tokens[0], "stoch") == 0) NetworkCLI_trainStochastic(&stochasticBP, &com);
			else if (strcmp(com.tokens[0], "minWeights") == 0) NetworkCLI_reportMinWeights(&com, &onlineBP, &stochasticBP);
			else if (strcmp(com.tokens[0], "loadWeights") == 0) NetworkCLI_loadMinWeights(&com, net, &onlineBP, &stochasticBP);
			else if (strcmp(com.tokens[0], "setWeights") == 0) NetworkCLI_setWeights(net, &com);
			else if (strcmp(com.tokens[0], "randomWeights") == 0) NetworkCLI_randomWeights(net);
			else if (strcmp(com.tokens[0], "") == 0) continue;
			else printf("Unknown command: %s\n", com.tokens[0]);
		}

		//cleanup
		BPTrainer_deinit(&onlineBP);
		BPTrainer_deinit(&stochasticBP);
		free(com.tokens);
	}
