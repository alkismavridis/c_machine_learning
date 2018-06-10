#include "Network.h"
#include<math.h>



//LINEAR
	NeuronUnit NeuronActivator_linear(NeuronUnit x) {
		return x;
	}

	NeuronUnit NeuronActivator_linearDerivative(NeuronUnit x) {
		return 1;
	}

//RELU
	NeuronUnit NeuronActivator_relu(NeuronUnit x) {
		return x>0 ? x : 0;
	}

	NeuronUnit NeuronActivator_reluDerivative(NeuronUnit x) {
		return x>0 ? 1 : 0;
	}

//LEAKY RELU
	#define NeuronActivator_REALU_LEAK_FACTOR 0.001
	NeuronUnit NeuronActivator_leakyRelu(NeuronUnit x) {
		return x>0 ? x : NeuronActivator_REALU_LEAK_FACTOR * x;
	}

	NeuronUnit NeuronActivator_leakyReluDerivative(NeuronUnit x) {
		return x>0 ? 1 : NeuronActivator_REALU_LEAK_FACTOR;
	}


//SIGMOID
	NeuronUnit NeuronActivator_sigmoid(NeuronUnit x) {
		return 1/(1+ exp(-x));
	}

	NeuronUnit NeuronActivator_sigmoidDerivative(NeuronUnit x) {
		NeuronUnit act = 1/(1+ exp(-x));
		return act * (1 - act);
	}


//TANH
	NeuronUnit NeuronActivator_tanh(NeuronUnit x) {
		return tanh(x);
	}

	NeuronUnit NeuronActivator_tanhDerivative(NeuronUnit x) {
		NeuronUnit th = (NeuronUnit) tanh(x);
		return 1 - th*th;
	}


//SETUP
	void NeuronActivator_init(NeuronActivator *this, NetworkLayerStructure *str) {
		switch (str->activatorType) {
			case NeuronActivator_RELU:
				this->inToOut = *NeuronActivator_relu;
				this->inToDerivative = *NeuronActivator_reluDerivative;
				break;

			case NeuronActivator_LEAKY_RELU:
				this->inToOut = *NeuronActivator_leakyRelu;
				this->inToDerivative = *NeuronActivator_leakyReluDerivative;
				break;

			case NeuronActivator_SIGMOID:
				this->inToOut = *NeuronActivator_sigmoid;
				this->inToDerivative = *NeuronActivator_sigmoidDerivative;
				break;

			case NeuronActivator_LINEAR:
				this->inToOut = *NeuronActivator_linear;
				this->inToDerivative = *NeuronActivator_linearDerivative;
				break;

			case NeuronActivator_TANH:
				this->inToOut = *NeuronActivator_tanh;
				this->inToDerivative = *NeuronActivator_tanhDerivative;
				break;

			case NeuronActivator_CUSTOM:
				this->inToOut = str->activationFunc;
				this->inToDerivative = str->activationDerivative;
				break;

			default:
				this->inToOut = NULL;
				this->inToDerivative = NULL;
				break;
		}
	}
