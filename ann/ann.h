/*

	This is a header file for Artificial Neural Network

*/

#pragma once
#include <cmath>
#include <iostream>

/// <summary>
/// Scales value between -1 and 1.
/// f(x) = x / (1 + abs(x))
/// </summary>
/// <param name="x"> - Value to scale</param>
/// <returns>Normalized value</returns>
float normalize(float x)
{
	return x / (1 + abs(x));
}

/// <summary>
/// Atrificial Neural Network
/// </summary>
/// <param name="input_amount"> - Amount of input_amount neurons</param>
/// <param name="output_amount"> - Amount of output_amount neurons</param>
/// <param name="hidden_amount"> - Amount of hidden layers</param>
/// <param name="hidden_size"> - Size of hidden layers</param>
template <unsigned int input_amount, unsigned int output_amount, unsigned int hidden_amount, unsigned int hidden_size>
struct ann
{
	/// <summary>
	/// 0 - no learning.
	/// 1 - full learning.
	/// 2 - 200% learning.
	/// and so on...
	/// </summary>
	float learning_rate = 0.f;

	struct
	{
		float input		[input_amount];
		float hidden	[hidden_amount][hidden_size]; // Dimentions are reversed for faster access
		float output	[output_amount];
	} neurons;

	struct
	{
		/// <summary>
		/// Layer between input_amount layer and first hidden layer. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - position in INPUT layer</param>
		/// <param name="Second dimention"> - position in FIRST HIDDEN layer</param>
		float first_layer	[hidden_size]		[input_amount];

		/// <summary>
		/// Weights between hidden layers. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - number between which hidden layers (0 - between first hidden and second hidden)</param>
		/// <param name="Second dimention"> - position in NEXT HIDDEN layer</param>
		/// <param name="Third dimention"> - position in PREVIOUS HIDDEN layer</param>
		float hidden_layers	[hidden_amount - 1]	[hidden_size]	[hidden_size];

		/// <summary>
		/// Layer between last hidden layer and output_amount layer. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - position in LAST HIDDEN layer layer</param>
		/// <param name="Second dimention"> - position in OUTPUT layer</param>
		float last_layer	[hidden_size]		[output_amount];
	} weights;

	ann();
	ann(const ann&) = default;

	void reset_neurons();
	void calc_output();
};

template<unsigned int input_amount, unsigned int output_amount, unsigned int hidden_amount, unsigned int hidden_size>
inline ann<input_amount, output_amount, hidden_amount, hidden_size>::ann()
{
	for (unsigned int i = 0; i < input_amount; i++)
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
	for (unsigned int i = 0; i < output_amount; i++)
	{
		neurons.output[i] = 0.f;
	}
}

template<unsigned int input_amount, unsigned int output_amount, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_amount, output_amount, hidden_amount, hidden_size>::reset_neurons()
{
	for (unsigned int i = 0; i < input_amount; i++)
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
	for (unsigned int i = 0; i < output_amount; i++)
	{
		neurons.output[i] = 0.f;
	}
}

template<unsigned int input_amount, unsigned int output_amount, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_amount, output_amount, hidden_amount, hidden_size>::calc_output()
{
	for (unsigned int i = 0; i < hidden_amount; i++) // Resetting hidden layers
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			neurons.hidden[i][j] = 0.f;
		}
	}
	for (unsigned int i = 0; i < output_amount; i++) // Resetting output_amount layer
	{
		neurons.output[i] = 0.f;
	}

	for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
	{
		for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < input_amount; prev_layer_neuron++)
		{
			neurons.hidden[0][next_layer_neuron] += neurons.input[prev_layer_neuron] * weights.first_layer[next_layer_neuron][prev_layer_neuron];
		}
		neurons.hidden[0][next_layer_neuron] = normalize(neurons.hidden[0][next_layer_neuron]);
		std::cout << "hidden[0][" << next_layer_neuron << "] = " << neurons.hidden[0][next_layer_neuron] << '\n';
	}

	for (unsigned int layer_number = 1; layer_number < hidden_amount; layer_number++)
	{
		for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
		{
			for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < hidden_size; prev_layer_neuron++)
			{
				neurons.hidden[layer_number][next_layer_neuron] += neurons.hidden[layer_number - 1][prev_layer_neuron] * weights.hidden_layers[layer_number - 1][next_layer_neuron][prev_layer_neuron];
			}
			neurons.hidden[layer_number][next_layer_neuron] = normalize(neurons.hidden[layer_number][next_layer_neuron]);
			std::cout << "hidden[" << layer_number << "][" << next_layer_neuron << "] = " << neurons.hidden[layer_number][next_layer_neuron] << '\n';
		}
	}

	for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
	{
		for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < output_amount; prev_layer_neuron++)
		{
			neurons.output[next_layer_neuron] += neurons.hidden[hidden_amount - 1][prev_layer_neuron] * weights.last_layer[next_layer_neuron][prev_layer_neuron];
		}
		neurons.output[next_layer_neuron] = normalize(neurons.output[next_layer_neuron]);
		std::cout << "output[" << next_layer_neuron << "] = " << neurons.output[next_layer_neuron] << '\n';
	}
}
