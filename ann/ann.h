/*

	This is a header file for Artificial Neural Network

	PRAISE THE CODE

*/

// #define ANN_DEBUG

// https://academy.yandex.ru/handbook/ml/article/metod-obratnogo-rasprostraneniya-oshibki

#pragma once
#include <cmath>
#include <fstream>
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include "D:\C++\Tools\eigen-3.4.0\Eigen\Eigen" // This is bs
#ifdef ANN_DEBUG
#include <iostream>
#endif // ANN_DEBUG

inline float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

/// <summary>
/// Calculates random value between `min` and `max`
/// </summary>
/// <param name="min"> - Lower bound of random value</param>
/// <param name="max"> - Upper bound of random value</param>
/// <returns>Random value between `min` and `max`</returns>
float rand_float(float min, float max)
{
	return min + static_cast<float>(rand()) / ((float(RAND_MAX) / (max - min)));
}

/// <summary>
/// Sample for teaching and testing Neural Network. `sample::input_size` should be equal to `ann:input_size`, just like `sample::output_size` should be equal to `ann::output_size`.
/// </summary>
/// <param name="input_size"> - input neurons</param>
/// <param name="output_size"> - expected results</param>
template <unsigned int input_size, unsigned int output_size>
struct sample
{
	Eigen::Vector<float, input_size>	input;
	Eigen::Vector<float, output_size>	output;
};

/// <summary>
/// Atrificial Neural Network. Uses sigmoid as activation function.
/// </summary>
/// <param name="input_size"> - Size of input layer</param>
/// <param name="output_size"> - Size of output layer</param>
/// <param name="hidden_amount"> - Amount of hidden layers</param>
/// <param name="hidden_size"> - Size of hidden layers</param>
template <unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
struct ann
{
	/// <summary>
	/// 0 - no learning.
	/// 1 - full learning.
	/// 2 - 200% learning.
	/// And so on...
	/// </summary>
	float learning_rate = 0.0f;

	struct
	{
		Eigen::Vector<float, input_size>	input;
		Eigen::Vector<float, hidden_size>	hidden[hidden_amount];
		Eigen::Vector<float, output_size>	output;
	} neurons;

	struct
	{
		Eigen::Vector<float, hidden_size>	hidden[hidden_amount];
		Eigen::Vector<float, output_size>	output;
	} biases;

	struct
	{
		Eigen::Matrix<float, hidden_size, input_size>	first_layer;
		Eigen::Matrix<float, hidden_size, hidden_size>	hidden_layers[hidden_amount - 1];
		Eigen::Matrix<float, output_size, hidden_size>	last_layer;
	} weights;

	ann();
	ann(const ann&) = default;

	bool load(std::string path);

	bool save(std::string path) const;

	/// <summary>
	/// Calculates output layer based on input layer and weights
	/// </summary>
	void forward_propagation();

	/// <summary>
	/// Sets all neurons (input, hidden, output) to zero
	/// </summary>
	void reset_neurons();

	/// <summary>
	/// Sets all input neurons to random values between min and max
	/// </summary>
	/// <param name="min"> - Lower bound of random values</param>
	/// <param name="max"> - Upper bound of random values</param>
	void set_input_to_rand(float min, float max);

	/// <summary>
	/// Sets all weights to random values between min and max
	/// </summary>
	/// <param name="min"> - Lower bound of random values</param>
	/// <param name="max"> - Upper bound of random values</param>
	void set_weights_to_rand(float min, float max);

	/// <summary>
	/// Cost function between `neurons.output` and `target`.
	/// </summary>
	/// <param name="target"> - target results</param>
	/// <returns>Sum of (target - out)^2</returns>
	float calc_cost(const Eigen::Vector<float, output_size>& target) const;

	/// <summary>
	/// Backpropagates the error
	/// </summary>
	/// <param name="target_results"> - Expected results that ANN should aim for</param>
	void backpropagation(const Eigen::Vector<float, output_size>& target);

	void learn(const sample<input_size, output_size>* samples, unsigned int amount);
};

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline ann<input_size, output_size, hidden_amount, hidden_size>::ann()
{
	for (unsigned int i = 0; i < input_size; i++)
	{
		neurons.input[i] = 0.f;
	}
	for (unsigned int i = 0; i < hidden_amount; i++)
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			neurons.hidden[i][j] = 0.f;
		}
	}
	for (unsigned int i = 0; i < output_size; i++)
	{
		neurons.output[i] = 0.f;
	}

	for (unsigned int i = 0; i < hidden_amount; i++)
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			biases.hidden[i][j] = 0.f;
		}
	}
	for (unsigned int i = 0; i < output_size; i++)
	{
		biases.output[i] = 0.f;
	}

	for (unsigned int i = 0; i < hidden_size; i++)
	{
		for (unsigned int j = 0; j < input_size; j++)
		{
			weights.first_layer(i, j) = 0;
		}
	}
	for (unsigned int layer_number = 0; layer_number < hidden_amount - 1; layer_number++)
	{
		for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
		{
			for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < hidden_size; prev_layer_neuron++)
			{
				weights.hidden_layers[layer_number](next_layer_neuron, prev_layer_neuron) = 0;
			}
		}
	}
	for (unsigned int i = 0; i < output_size; i++)
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			weights.last_layer(i, j) = 0;
		}
	}
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline bool ann<input_size, output_size, hidden_amount, hidden_size>::load(std::string path)
{
	std::ifstream fin(path, std::ios::binary);
	unsigned int temp1, temp2, temp3, temp4;

	if (!fin)
	{
		return false;
	}

	fin.read((char*)&temp1, sizeof(temp1));
	fin.read((char*)&temp2, sizeof(temp2));
	fin.read((char*)&temp3, sizeof(temp3));
	fin.read((char*)&temp4, sizeof(temp4));

	if (temp1 != input_size || temp2 != output_size || temp3 != hidden_amount || temp4 != hidden_size)
	{
		fin.close();
		return false;
	}

	fin.read((char*)this, sizeof(*this));

	fin.close();
	return true;
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline bool ann<input_size, output_size, hidden_amount, hidden_size>::save(std::string path) const
{
	std::ofstream fout(path, std::ios::binary);

	if (!fout)
	{
		return false;
	}

	unsigned int temp1 = input_size, temp2 = output_size, temp3 = hidden_amount, temp4 = hidden_size;

	fout.write((char*)&temp1, sizeof(temp1));
	fout.write((char*)&temp2, sizeof(temp2));
	fout.write((char*)&temp3, sizeof(temp3));
	fout.write((char*)&temp4, sizeof(temp4));
	fout.write((char*)this, sizeof(*this));

	fout.close();
	return true;
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_size, output_size, hidden_amount, hidden_size>::forward_propagation()
{
	for (unsigned int i = 0; i < hidden_amount; i++) // Resetting hidden layers
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			neurons.hidden[i][j] = 0.f;
		}
	}
	for (unsigned int i = 0; i < output_size; i++) // Resetting output layer
	{
		neurons.output[i] = 0.f;
	}

	neurons.hidden[0] = weights.first_layer * neurons.input;
	for (unsigned int i = 0; i < hidden_size; i++)
	{
		neurons.hidden[0][i] = sigmoid(neurons.hidden[0][i] + biases.hidden[0][i]);
	}

	for (unsigned int i = 1; i < hidden_amount; i++)
	{
		neurons.hidden[i] = weights.hidden_layers[i - 1] * neurons.hidden[i - 1];
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			neurons.hidden[i][j] = sigmoid(neurons.hidden[i][j] + biases.hidden[i][j]);
		}
	}

	neurons.output = weights.last_layer * neurons.hidden[hidden_size - 1];
	for (unsigned int i = 0; i < output_size; i++)
	{
		neurons.output[i] = sigmoid(neurons.output[i] + biases.output[i]);
	}
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_size, output_size, hidden_amount, hidden_size>::reset_neurons()
{
	for (unsigned int i = 0; i < input_size; i++)
	{
		neurons.input[i] = 0.f;
	}
	for (unsigned int i = 0; i < hidden_amount; i++)
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			neurons.hidden[i][j] = 0.f;
		}
	}
	for (unsigned int i = 0; i < output_size; i++)
	{
		neurons.output[i] = 0.f;
	}
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_size, output_size, hidden_amount, hidden_size>::set_input_to_rand(float min, float max)
{
	for (unsigned int i = 0; i < input_size; i++)
	{
		neurons.input[i] = rand_float(min, max);
	}
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_size, output_size, hidden_amount, hidden_size>::set_weights_to_rand(float min, float max)
{
	for (unsigned int i = 0; i < hidden_size; i++)
	{
		for (unsigned int j = 0; j < input_size; j++)
		{
			weights.first_layer(i, j) = rand_float(min, max);
		}
	}
	for (unsigned int layer_number = 0; layer_number < hidden_amount - 1; layer_number++)
	{
		for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
		{
			for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < hidden_size; prev_layer_neuron++)
			{
				weights.hidden_layers[layer_number](next_layer_neuron, prev_layer_neuron) = rand_float(min, max);
			}
		}
	}
	for (unsigned int i = 0; i < output_size; i++)
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			weights.last_layer(i, j) = rand_float(min, max);
		}
	}
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline float ann<input_size, output_size, hidden_amount, hidden_size>::calc_cost(const Eigen::Vector<float, output_size>& target) const
{
	float result = 0.0;

	for (unsigned int i = 0; i < output_size; i++)
	{
		result += (target[i] - neurons.output[i]) * (target[i] - neurons.output[i]);
	}

	return result;
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_size, output_size, hidden_amount, hidden_size>::backpropagation(const Eigen::Vector<float, output_size>& target)
{
	// TODO: optimise memory usage

	struct
	{
		Eigen::Vector<float, hidden_size>	hidden[hidden_amount];
		Eigen::Vector<float, output_size>	output;
	} derivatives;

	// calculation effect of Neurons

	for (unsigned int i = 0; i < output_size; i++) // output layer
	{
		derivatives.output[i] = (neurons.output[i] - target[i]) * neurons.output[i] * (1 - neurons.output[i]);
	}

	for (unsigned int i = 0; i < hidden_size; i++) // last hidden layer
	{
		derivatives.hidden[hidden_amount - 1][i] = 0;

		for (unsigned int j = 0; j < output_size; j++)
		{
			derivatives.hidden[hidden_amount - 1][i] += weights.last_layer(j, i) * derivatives.output[j];
		}

		derivatives.hidden[hidden_amount - 1][i] *= neurons.hidden[hidden_amount - 1][i] * (1 - neurons.hidden[hidden_amount - 1][i]);
	}

	for (unsigned int layer = hidden_amount - 1; layer > 0; layer--) // remaining hidden layers
	{
		for (unsigned int i = 0; i < hidden_size; i++)
		{
			derivatives.hidden[layer - 1][i] = 0;

			for (unsigned int j = 0; j < hidden_size; j++)
			{
				derivatives.hidden[layer - 1][i] += weights.hidden_layers[layer](j, i) * derivatives.hidden[layer][j];
			}

			derivatives.hidden[layer - 1][i] *= neurons.hidden[layer - 1][i] * (1 - neurons.hidden[layer - 1][i]);
		}
	}

	// changing weights

	for (unsigned int i = 0; i < output_size; i++) // last layer
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			weights.last_layer(i, j) -= learning_rate * derivatives.output[i] * neurons.hidden[hidden_amount - 1][j];
			//                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			//                                          ^ derivative of Weight
		}
	}

	for (unsigned int layer = hidden_amount - 2; layer > 0; layer--) // hidden layers
	{
		for (unsigned int i = 0; i < output_size; i++)
		{
			for (unsigned int j = 0; j < hidden_size; j++)
			{
				weights.hidden_layers[layer -1](i, j) -= learning_rate * derivatives.hidden[layer][i] * neurons.hidden[layer - 1][j];
				//                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
				//                                                       ^ derivative of Weight
			}
		}
	}

	for (unsigned int i = 0; i < hidden_size; i++) // first layer
	{
		for (unsigned int j = 0; j < input_size; j++)
		{
			weights.first_layer(i, j) -= learning_rate * derivatives.hidden[0][i] * neurons.input[j];
			//                                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			//                                           ^ derivative of Weight
		}
	}
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_size, output_size, hidden_amount, hidden_size>::learn(const sample<input_size, output_size>* samples, unsigned int amount)
{
	unsigned int step = sqrt(amount);

	for (unsigned int i = 0; i < amount; i++)
	{
		forward_propagation(samples[i].output);

		if (i % step == 0)
		{
			// test
		}
		else
		{
			backpropagation(samples[i].output);
		}
	}
}
