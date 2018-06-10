#pragma once
#include <stdio.h>

//TYPE DEFINITIONS
	typedef double NeuronUnit;
	typedef struct _Neuron Neuron;

	typedef struct {
		NeuronUnit weight;
		unsigned short int targetIndex;
		Neuron *target;
	} NeuronSynapse;


	struct _Neuron {
		NeuronUnit in;
		NeuronUnit out;
		NeuronUnit delta;

		unsigned int synapseCount;
		NeuronSynapse * synapses;
	};


	#define NeuronActivator_CUSTOM 0
	#define NeuronActivator_SIGMOID 1
	#define NeuronActivator_TANH 2
	#define NeuronActivator_LINEAR 3
	#define NeuronActivator_RELU 4
	#define NeuronActivator_LEAKY_RELU 5
	typedef struct {
		NeuronUnit (*inToOut)(NeuronUnit x);
		NeuronUnit (*inToDerivative)(NeuronUnit x);
	} NeuronActivator;


	#define NetworkLayer_FULLY_CONNECTED 1
	#define NetworkLayer_INDIVIDUAL 2
	#define NetworkLayer_OUTPUT 3
	typedef struct {
		unsigned short int neuronCount;
		Neuron * neurons;
		Neuron bias;
		NeuronActivator activator;

		unsigned int synapseCount;
	} NetworkLayer;


	typedef struct {
		unsigned short int layerCount;
		NetworkLayer* layers;

		unsigned short int neuronCount;
		unsigned long int synapseCount;
	} NeuralNetwork;



//STRUCTURES. Those structs store information on how to initialize a Network.
	typedef struct {
		unsigned char connectionType;
		int** neurons; //NULL terminated. Each one of them is -1 terminated
		int* bias;		//-1 terminated
		unsigned char activatorType;

		//use thses only in case of activatorType == NeuronActivator_CUSTOM
		NeuronUnit (*activationFunc)(NeuronUnit x);
		NeuronUnit (*activationDerivative)(NeuronUnit x);

		unsigned short int neuronCount;
	} NetworkLayerStructure;

	typedef struct {
		NetworkLayerStructure* layers; //This array must have a LayerStructure with connetcionType NetworkLayer_OUTPUT as last element. This indicates the end.
	} NeuralNetworkStructure;




//Neuron functions
	void Neuron_init(Neuron* this, int* synapses);
	void Neuron_deinit(Neuron* this);

	void Neuron_bindForward(Neuron* this, NetworkLayer * nextLayer);
	void Neuron_bindBackward(Neuron* this, NetworkLayer * prevLayer);

	void Neuron_randomSynapses(Neuron* this);
	void Neuron_loadSynapseWeights(Neuron* this, NeuronUnit* weights);
	void Neuron_saveSynapseWeights(Neuron* this, NeuronUnit* buffer);
	void Neuron_adjustWeights(Neuron* this, NeuronUnit* offsets);
	void Neuron_reset(Neuron* this);

	void Neuron_fire(Neuron* this);
	void Neuron_activate(Neuron* this, NeuronUnit (*activationFunction)(NeuronUnit));
	void Neuron_calculateDeltaFromErrorDerivative(Neuron* this, NeuronUnit (*activationDerivative)(NeuronUnit), NeuronUnit errorDerivative);
	void Neuron_saveGradient(Neuron* this, NeuronUnit (*activationDerivative)(NeuronUnit), NeuronUnit* grad);
	void Neuron_addToGradient(Neuron* this,NeuronUnit (*activationDerivative)(NeuronUnit), NeuronUnit* grad);



//NeuronActivator functions
	void NeuronActivator_init(NeuronActivator *this, NetworkLayerStructure *str);
	NeuronUnit NeuronActivator_linear(NeuronUnit x);
	NeuronUnit NeuronActivator_linearDerivative(NeuronUnit x);
	NeuronUnit NeuronActivator_sigmoid(NeuronUnit x);
	NeuronUnit NeuronActivator_sigmoidDerivative(NeuronUnit x);


//NetworkLayer functions
	void NetworkLayer_initIndividual(NetworkLayer* this, NetworkLayerStructure* str);
	void NetworkLayer_initFullyConnected(NetworkLayer* this, NetworkLayerStructure* str, unsigned short int nextLayerNeuronCount);
	unsigned short int NetworkLayer_getNeuronCount(NetworkLayerStructure* str);
	void NetworkLayer_initOutput(NetworkLayer* this, NetworkLayerStructure* str);
	void NetworkLayer_deinit(NetworkLayer* this);

	void NetworkLayer_bindForward(NetworkLayer* this, NetworkLayer* next);

	void NetworkLayer_randomSynapses(NetworkLayer* this);
	void NetworkLayer_loadSynapseWeights(NetworkLayer* this, NeuronUnit* weights);
	void NetworkLayer_saveSynapseWeights(NetworkLayer* this, NeuronUnit* buffer);
	void NetworkLayer_adjustWeights(NetworkLayer* this, NeuronUnit* offsets);


	void NetworkLayer_reset(NetworkLayer* this);
	void NetworkLayer_fire(NetworkLayer* this);
	void NetworkLayer_activate(NetworkLayer *this);
	void NetworkLayer_calculateDeltaFromErrorDerivatives(NetworkLayer* this, NeuronUnit *errorDerivatives);
	void NetworkLayer_saveGradient(NetworkLayer* this, NeuronUnit* grad);
	void NetworkLayer_addToGradient(NetworkLayer* this, NeuronUnit* grad);



//NeuralNetwork functions
	void NeuralNetwork_init(NeuralNetwork* this, NeuralNetworkStructure* s);
	void NeuralNetwork_deinit(NeuralNetwork* this);
	void NeuralNetwork_randomSynapses(NeuralNetwork* this);
	void NeuralNetwork_loadSynapseWeights(NeuralNetwork* this, NeuronUnit* weights);
	void NeuralNetwork_saveSynapseWeights(NeuralNetwork* this, NeuronUnit* buffer);
	void NeuralNetwork_adjustWeights(NeuralNetwork* this, NeuronUnit* offsets);

	NeuronUnit NeuralNetwork_activation(NeuronUnit x);
	NeuronUnit NeuralNetwork_activationDerivative(NeuronUnit x);

	void NeuralNetwork_predict(NeuralNetwork * this);
	void NeuralNetwork_saveGradient(NeuralNetwork *this, NeuronUnit *errorDerivatives, NeuronUnit* grad);
	void NeuralNetwork_addToGradient(NeuralNetwork *this, NeuronUnit *errorDerivatives, NeuronUnit* grad);





//NeuralNetworkStructure functions
	void NeuralNetworkStructure_deinit();
