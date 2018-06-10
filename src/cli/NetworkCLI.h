#pragma once

#include "../network/Network.h"
#include "../train/NetworkTrain.h"

void NetworkCLI_start(
	NeuralNetwork *net,
	TrainDataProvider *provider,
	void (*onlineBPErrorFunction)(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients),
	void (*stochasticBPErrorFunction)(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients));
