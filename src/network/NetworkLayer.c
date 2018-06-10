#include "Network.h"
#include <stdlib.h>
#include <string.h>
#include<stdio.h>
#include<math.h>

//LIFE CIRCLE
	void NetworkLayer_initIndividual(NetworkLayer* this, NetworkLayerStructure* str) {
		unsigned short int neuronCount = NetworkLayer_getNeuronCount(str);

		this->neurons = (Neuron*) malloc(neuronCount * sizeof(Neuron));

		unsigned int totalSynapseCount = 0;
		for (unsigned int i = neuronCount; i--; ) {
			Neuron_init(this->neurons + i, str->neurons[i]);
			totalSynapseCount += this->neurons[i].synapseCount;
		}

		Neuron_init(&(this->bias), str->bias);
		totalSynapseCount += this->bias.synapseCount;
		this->bias.out = 1;

		this->neuronCount = neuronCount;
		this->synapseCount = totalSynapseCount;
		NeuronActivator_init(&(this->activator), str);
	}

	void NetworkLayer_initFullyConnected(NetworkLayer* this, NetworkLayerStructure* str, unsigned short int nextLayerNeuronCount) {
		this->neuronCount = str->neuronCount;
		this->neurons = (Neuron*) malloc(this->neuronCount * sizeof(Neuron));


		//initialize synapse array
		int arrOfNeurons[this->neuronCount+1];
		for(unsigned short int i = 0; i < nextLayerNeuronCount; i++) {
			arrOfNeurons[i] = i;
		}
		arrOfNeurons[nextLayerNeuronCount] = -1;


		unsigned int totalSynapseCount = 0;
		for (unsigned int i = this->neuronCount; i--; ) {
			Neuron_init(this->neurons + i, arrOfNeurons);
			totalSynapseCount += this->neurons[i].synapseCount;
		}

		Neuron_init(&(this->bias), arrOfNeurons);
		totalSynapseCount += this->bias.synapseCount;
		this->bias.out = 1;

		this->synapseCount = totalSynapseCount;
		NeuronActivator_init(&(this->activator), str);
	}

	void NetworkLayer_initOutput(NetworkLayer* this, NetworkLayerStructure* str) {
		this->neuronCount = str->neuronCount;
		this->neurons = (Neuron*) malloc(this->neuronCount * sizeof(Neuron));


		//initialize synapse "array"
	    int arrOfNeurons = -1;


		for (unsigned int i = this->neuronCount; i--; ) {
			Neuron_init(this->neurons + i, &arrOfNeurons);
		}

		Neuron_init(&(this->bias), &arrOfNeurons);
		this->bias.out = 1;

		this->synapseCount = 0;
		NeuronActivator_init(&(this->activator), str);
	}

	void NetworkLayer_deinit(NetworkLayer* this) {
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			Neuron_deinit(&this->neurons[i]);
			if (i==0) break;
		}
		Neuron_deinit(&this->bias);
		free(this->neurons);
	}



//GETTERS
	unsigned short int NetworkLayer_getNeuronCount(NetworkLayerStructure* str) {
		switch (str->connectionType) {
			case NetworkLayer_INDIVIDUAL: {
				unsigned short int ret = 0;
				while(str->neurons[ret] != NULL) ret++;
				return ret;
			}

			default:
				return str->neuronCount;
		}
	}



//STRUCTURE SETUP
	void NetworkLayer_bindForward(NetworkLayer* this, NetworkLayer* next) {
		if (this->neuronCount==0) return;

		for (unsigned short int i = this->neuronCount; i--;) {
			Neuron_bindForward(&this->neurons[i], next);
		}

		Neuron_bindForward(&this->bias, next);
	}


//STATE SETUP
	void NetworkLayer_randomSynapses(NetworkLayer* this) {
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			Neuron_randomSynapses(&this->neurons[i]);
			if (i==0) break;
		}

		Neuron_randomSynapses(&this->bias);
	}

	void NetworkLayer_loadSynapseWeights(NetworkLayer* this, NeuronUnit* weights) {
		NeuronUnit *neuronWeights = weights + (this->synapseCount) - this->bias.synapseCount;
		Neuron_loadSynapseWeights(&(this->bias), neuronWeights);

		if (this->neuronCount<=0) return;
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			neuronWeights -= this->neurons[i].synapseCount;
			Neuron_loadSynapseWeights(&(this->neurons[i]), neuronWeights);
			if (i==0) break;
		}
	}

	void NetworkLayer_saveSynapseWeights(NetworkLayer* this, NeuronUnit* buffer) {
		NeuronUnit *neuronBuffer = buffer + (this->synapseCount) - this->bias.synapseCount;
		Neuron_saveSynapseWeights(&(this->bias), neuronBuffer);

		if (this->neuronCount<=0) return;
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			neuronBuffer -= this->neurons[i].synapseCount;
			Neuron_saveSynapseWeights(&(this->neurons[i]), neuronBuffer);
			if (i==0) break;
		}
	}

	void NetworkLayer_adjustWeights(NetworkLayer* this, NeuronUnit* offsets) {
		NeuronUnit *neuronOffsets = offsets + (this->synapseCount) - this->bias.synapseCount;
		Neuron_adjustWeights(&(this->bias), neuronOffsets);

		if (this->neuronCount<=0) return;
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			neuronOffsets -= this->neurons[i].synapseCount;
			Neuron_adjustWeights(&(this->neurons[i]), neuronOffsets);
			if (i==0) break;
		}
	}




//LAYER OPERATIONS
	void NetworkLayer_reset(NetworkLayer* this) {
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			Neuron_reset(&this->neurons[i]);
			if (i==0) break;
		}
	}

	void NetworkLayer_fire(NetworkLayer* this) {
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			Neuron_fire(&this->neurons[i]);
			if (i==0) break;
		}

		Neuron_fire(&this->bias);
	}

	void NetworkLayer_activate(NetworkLayer* this) {
		NeuronUnit (*activationFunction)(NeuronUnit) = this->activator.inToOut;
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			Neuron_activate(&this->neurons[i], activationFunction);
			if (i==0) break;
		}
	}

	void NetworkLayer_calculateDeltaFromErrorDerivatives(NetworkLayer* this, NeuronUnit *errorDerivatives) {
		NeuronUnit (*activationDerivative)(NeuronUnit) = this->activator.inToDerivative;
		for (unsigned short int i = this->neuronCount-1; 1; --i) {
			Neuron_calculateDeltaFromErrorDerivative(&this->neurons[i], activationDerivative, errorDerivatives[i]);
			if (i==0) break;
		}
	}

	void NetworkLayer_saveGradient(NetworkLayer* this, NeuronUnit* grad) {
		NeuronUnit *neuronBasis = grad + (this->synapseCount) - this->bias.synapseCount;
		NeuronUnit (*activationDerivative)(NeuronUnit) = this->activator.inToDerivative;
		Neuron_saveGradient(&this->bias, NULL, neuronBasis);

		for (unsigned short int i = this->neuronCount; i--;) {
			neuronBasis -= this->neurons[i].synapseCount;
			Neuron_saveGradient(&this->neurons[i], activationDerivative, neuronBasis);
		}
	}

	void NetworkLayer_addToGradient(NetworkLayer* this, NeuronUnit* grad) {
		NeuronUnit *neuronBasis = grad + (this->synapseCount) - this->bias.synapseCount;
		NeuronUnit (*activationDerivative)(NeuronUnit) = this->activator.inToDerivative;
		Neuron_addToGradient(&this->bias, NULL, neuronBasis);

		for (unsigned short int i = this->neuronCount; i--;) {
			neuronBasis -= this->neurons[i].synapseCount;
			Neuron_addToGradient(&this->neurons[i], activationDerivative, neuronBasis);
		}
	}
