#pragma once
#include "../network/Network.h"


//FORWARD DECLARATIONS
	typedef struct _TrainDataProvider TrainDataProvider;




//TYPE DEFINITIONS
	struct _TrainDataProvider {
		unsigned int counter;
		unsigned int maxResults;
		NeuronUnit *expected;
		char (*provideInput)(TrainDataProvider* this, NeuralNetwork* net); //returns 0 if no data are available
	};


	typedef struct {
		NeuronUnit* weights;
		NeuronUnit error;
	} ErrorPoint;

	typedef struct {
		//props
		NeuralNetwork* network;
		TrainDataProvider *provider;
		void (*errorUpdater)(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients);


		//state
		char isTraining;
		ErrorPoint start;
		ErrorPoint minimum;
	} BPTrainer;



//FUNCTIONS

	void TrainDataProvider_init(
		TrainDataProvider* this,
		char (*provideInput)(TrainDataProvider* this, NeuralNetwork* net),
		unsigned short int outputCount,
		unsigned int maxResults);
	void TrainDataProvider_deinit(TrainDataProvider* this);
	void TrainDataProvider_reset(TrainDataProvider* this, unsigned int maxReslts);


	void BPTrainer_init(
		BPTrainer* this,
		NeuralNetwork* network,
		TrainDataProvider *provider,
		void (*errorUpdater)(NeuralNetwork* net, NeuronUnit* expected, NeuronUnit* errorValue, NeuronUnit* errorGradients));
	void BPTrainer_deinit(BPTrainer* this);

	void BPTrainer_trainOnline(BPTrainer* this, NeuronUnit learningRate, NeuronUnit momentum, char debug);
	void BPTrainer_trainStochastic(BPTrainer* this, unsigned int updateEvery, NeuronUnit learningRate, NeuronUnit momentum, char debug);
	void BPTrainer_stopTraining(BPTrainer* this);

	void BPTrainer_saveState(BPTrainer* this, FILE f);
	void BPTrainer_loadState(BPTrainer* this, FILE f);
