/*

	This is a header file for Artificial Neural Network

*/

// #define ANN_DEBUG

#pragma once
#include <cmath>
#ifdef ANN_DEBUG
#include <iostream>
#endif // ANN_DEBUG


/// <summary>
/// Scales value between -1 and 1.
/// f(x) = x / (1 + abs(x))
/// </summary>
/// <param name="x"> - Value to scale</param>
/// <returns>Normalized value</returns>
float soft_max(float x)
{
	return x / (1 + abs(x));
}

/// <summary>
/// Derivative of `soft_max`. Used for backpropagation.
/// f(x) = 1 / (2 * abs(x) + x * x + 1)
/// </summary>
/// <param name="x"> - Value to scale</param>
/// <returns>Normalized value</returns>
float soft_max_derivative(float x)
{
	return 1 / (2 * abs(x) + x * x + 1);
}

/// <summary>
/// Calculates random value between `min` and `max`
/// </summary>
/// <param name="min"> - Lower bound of random value</param>
/// <param name="max"> - Upper bound of random value</param>
/// <returns>Random value between `min` and `max`</returns>
float rand_float(float min, float max)
{
	return min + static_cast<float>(rand()) / (static_cast<float>(float(RAND_MAX) / (max - min)));
}

/// <summary>
/// Atrificial Neural Network
/// </summary>
/// <param name="input_size"> - Size of input layer</param>
/// <param name="output_size"> - Size of output layer</param>
/// <param name="hidden_amount"> - Amount of hidden layers</param>
/// <param name="hidden_size"> - Size of hidden layers</param>
/// <param name="normalization_func"> - Function that will be used to scale values in neurons</param>
/// <param name="derivative_of_norm_func"> - Derivative of normalization function that will be used in backpropagation</param>
template <unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
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
		float input		[input_size];
		float hidden	[hidden_amount][hidden_size]; // Dimentions are reversed for faster access
		float output	[output_size];
	} neurons;

	struct
	{
		/// <summary>
		/// Layer between input_size layer and first hidden layer. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - position in INPUT layer</param>
		/// <param name="Second dimention"> - position in FIRST HIDDEN layer</param>
		float first_layer	[hidden_size]		[input_size];

		/// <summary>
		/// Weights between hidden layers. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - number between which hidden layers (0 - between first hidden and second hidden)</param>
		/// <param name="Second dimention"> - position in NEXT HIDDEN layer</param>
		/// <param name="Third dimention"> - position in PREVIOUS HIDDEN layer</param>
		float hidden_layers	[hidden_amount - 1]	[hidden_size]	[hidden_size];

		/// <summary>
		/// Layer between last hidden layer and output_size layer. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - position in LAST HIDDEN layer layer</param>
		/// <param name="Second dimention"> - position in OUTPUT layer</param>
		float last_layer	[hidden_size]		[output_size];
	} weights;

	ann();
	ann(const ann&) = default;

	/// <summary>
	/// Calculates output layer based on input layer and weights
	/// </summary>
	void calc_output();

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
	/// Backpropagates the error
	/// </summary>
	/// <param name="expected_results"> - Expected results that ANN should aim for</param>
	void backpropagation(float expected_results[output_size]);
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

	for (unsigned int i = 0; i < hidden_size; i++)
	{
		for (unsigned int j = 0; j < input_size; j++)
		{
			weights.first_layer[i][j] = 0;
		}
	}
	for (unsigned int layer_number = 0; layer_number < hidden_amount - 1; layer_number++)
	{
		for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
		{
			for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < hidden_size; prev_layer_neuron++)
			{
				weights.hidden_layers[layer_number][next_layer_neuron][prev_layer_neuron] = 0;
			}
		}
	}
	for (unsigned int i = 0; i < output_size; i++)
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			weights.last_layer[i][j] = 0;
		}
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
inline void ann<input_size, output_size, hidden_amount, hidden_size>::calc_output()
{
	for (unsigned int i = 0; i < hidden_amount; i++) // Resetting hidden layers
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			neurons.hidden[i][j] = 0.f;
		}
	}
	for (unsigned int i = 0; i < output_size; i++) // Resetting output_size layer
	{
		neurons.output[i] = 0.f;
	}

	for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
	{
		for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < input_size; prev_layer_neuron++)
		{
			neurons.hidden[0][next_layer_neuron] += neurons.input[prev_layer_neuron] * weights.first_layer[next_layer_neuron][prev_layer_neuron];
		}
		neurons.hidden[0][next_layer_neuron] = normalization_func(neurons.hidden[0][next_layer_neuron]);
#ifdef ANN_DEBUG
		std::cout << "hidden[0][" << next_layer_neuron << "] = " << neurons.hidden[0][next_layer_neuron] << '\n';
#endif // ANN_DEBUG
	}

	for (unsigned int layer_number = 1; layer_number < hidden_amount; layer_number++)
	{
		for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
		{
			for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < hidden_size; prev_layer_neuron++)
			{
				neurons.hidden[layer_number][next_layer_neuron] += neurons.hidden[layer_number - 1][prev_layer_neuron] * weights.hidden_layers[layer_number - 1][next_layer_neuron][prev_layer_neuron];
			}
			neurons.hidden[layer_number][next_layer_neuron] = normalization_func(neurons.hidden[layer_number][next_layer_neuron]);
#ifdef ANN_DEBUG
			std::cout << "hidden[" << layer_number << "][" << next_layer_neuron << "] = " << neurons.hidden[layer_number][next_layer_neuron] << '\n';
#endif // ANN_DEBUG
		}
	}

	for (unsigned int next_layer_neuron = 0; next_layer_neuron < output_size; next_layer_neuron++)
	{
		for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < hidden_size; prev_layer_neuron++)
		{
			neurons.output[next_layer_neuron] += neurons.hidden[hidden_amount - 1][prev_layer_neuron] * weights.last_layer[next_layer_neuron][prev_layer_neuron];
		}
		neurons.output[next_layer_neuron] = normalization_func(neurons.output[next_layer_neuron]);
#ifdef ANN_DEBUG
		std::cout << "output[" << next_layer_neuron << "] = " << neurons.output[next_layer_neuron] << '\n';
#endif // ANN_DEBUG
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
inline void ann<input_size, output_size, hidden_amount, hidden_size>::set_weights_to_rand(float min, float max)
{
	for (unsigned int i = 0; i < hidden_size; i++)
	{
		for (unsigned int j = 0; j < input_size; j++)
		{
			weights.first_layer[i][j] = rand_float(min, max);
		}
	}
	for (unsigned int layer_number = 0; layer_number < hidden_amount; layer_number++)
	{
		for (unsigned int next_layer_neuron = 0; next_layer_neuron < hidden_size; next_layer_neuron++)
		{
			for (unsigned int prev_layer_neuron = 0; prev_layer_neuron < hidden_size; prev_layer_neuron++)
			{
				weights.hidden_layers[layer_number][next_layer_neuron][prev_layer_neuron] = rand_float(min, max);
			}
		}
	}
	for (unsigned int i = 0; i < output_size; i++)
	{
		for (unsigned int j = 0; j < hidden_size; j++)
		{
			weights.last_layer[i][j] = rand_float(min, max);
		}
	}
}

template<unsigned int input_size, unsigned int output_size, unsigned int hidden_amount, unsigned int hidden_size>
inline void ann<input_size, output_size, hidden_amount, hidden_size>::backpropagation(float expected_results[output_size])
{

}
