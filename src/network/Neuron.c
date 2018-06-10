#include "Network.h"
#include <stdlib.h>

//LIFE CIRCLE
	void Neuron_init(Neuron* this, int* synapses) {
		unsigned short int synapseCount = 0;
		while(synapses[synapseCount] != -1) synapseCount++;

		this->synapseCount = synapseCount;
		this->synapses = (NeuronSynapse*) malloc(synapseCount * sizeof(NeuronSynapse));

		if (synapseCount!=0) {
			for (unsigned int i = synapseCount; i--; ) this->synapses[i].targetIndex = (unsigned short int)synapses[i];
		}
	}

	void Neuron_deinit(Neuron* this) {
		free(this->synapses);
	}


//STRUCTURE SETUP
	void Neuron_bindForward(Neuron* this, NetworkLayer * nextLayer) {
		if (this->synapseCount==0) return;

		for (unsigned short int i = this->synapseCount; i--;) {
			NeuronSynapse *syn = this->synapses + i;
			syn->target = &nextLayer->neurons[ syn->targetIndex ];
		}
	}


//STATE SETUP
	void Neuron_randomSynapses(Neuron* this) {
		if (this->synapseCount==0) return;

		for (unsigned short int i = this->synapseCount; i--;) {
			this->synapses[i].weight = (NeuronUnit)rand() / ((NeuronUnit)RAND_MAX) * 2 - 1;
		}
	}

	void Neuron_loadSynapseWeights(Neuron* this, NeuronUnit* weights) {
		if (this->synapseCount==0) return;

		NeuronSynapse* synapses = this->synapses;
		for (unsigned short int i = this->synapseCount; i--;) synapses[i].weight = weights[i];
	}

	void Neuron_saveSynapseWeights(Neuron* this, NeuronUnit* buffer) {
		if (this->synapseCount==0) return;

		NeuronSynapse* synapses = this->synapses;
		for (unsigned short int i = this->synapseCount; i--;) buffer[i] = synapses[i].weight;
	}

	void Neuron_adjustWeights(Neuron* this, NeuronUnit* offsets) {
		if (this->synapseCount==0) return;

		NeuronSynapse* synapses = this->synapses;
		for (unsigned short int i = this->synapseCount; i--;) synapses[i].weight += offsets[i];
	}


//NEURON OPERATIONS
	void Neuron_reset(Neuron* this) {
		this->in = 0;
	}

	void Neuron_activate(Neuron* this, NeuronUnit (*activationFunction)(NeuronUnit)) {
		this->out = activationFunction(this->in);
	}

	void Neuron_fire(Neuron* this) {
		if (this->synapseCount==0) return;
		for (unsigned short int i = this->synapseCount; i--;) {
			NeuronSynapse *syn = this->synapses + i;
			syn->target->in += syn->weight * this->out;
		}
	}

	void Neuron_calculateDeltaFromErrorDerivative(Neuron* this, NeuronUnit (*activationDerivative)(NeuronUnit), NeuronUnit errorDerivative) {
		this->delta = activationDerivative( this->in ) * errorDerivative;
	}

	void Neuron_saveGradient(Neuron* this, NeuronUnit (*activationDerivative)(NeuronUnit), NeuronUnit* grad) {
		if (this->synapseCount == 0) return;
		NeuronSynapse* synapses = this->synapses;
		NeuronUnit output = this->out;

		if (activationDerivative==NULL) { //skip delta calculation
			for (unsigned short int i = this->synapseCount; i--;) {
				*(grad + i) = output * synapses[i].target->delta;
			}
		} else {
			NeuronUnit sum = 0;
			for (unsigned short int i = this->synapseCount; i--;) {
				NeuronSynapse *syn = synapses + i;
				sum += syn->weight * syn->target->delta;
				*(grad + i) = output * syn->target->delta;
			}

			this->delta = activationDerivative( this->in ) * sum;
		}
	}

	void Neuron_addToGradient(Neuron* this, NeuronUnit (*activationDerivative)(NeuronUnit), NeuronUnit* grad) {
		if (this->synapseCount == 0) return;
		NeuronSynapse* synapses = this->synapses;
		NeuronUnit output = this->out;

		if (activationDerivative==NULL) { //skip delta calculation
			for (unsigned short int i = this->synapseCount; i--;) {
				*(grad + i) += output * synapses[i].target->delta;
			}
		} else {
			NeuronUnit sum = 0;
			for (unsigned short int i = this->synapseCount; i--;) {
				NeuronSynapse *syn = synapses + i;
				sum += syn->weight * syn->target->delta;
				*(grad + i) += output * syn->target->delta;
			}

			this->delta = activationDerivative( this->in ) * sum;
		}
	}
