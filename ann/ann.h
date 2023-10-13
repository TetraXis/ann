/*

	This is a header for Artificial Neural Network

*/

#pragma once

/// <summary>
/// Atrificial Neural Network
/// </summary>
/// <param name="input"> - Amount of input neurons</param>
/// <param name="output"> - Amount of output neurons</param>
/// <param name="hidden_amount"> - Amount of hidden layers</param>
/// <param name="hidden_size"> - Size of hidden layers</param>
template <unsigned int input, unsigned int output, unsigned int hidden_amount, unsigned int hidden_size>
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
		float input		[input];
		float hidden	[hidden_amount][hidden_size]; // Dimentions are reversed for faster access
		float output	[output];
	} neurons;

	struct
	{
		/// <summary>
		/// Layer between input layer and first hidden layer. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - position in INPUT layer</param>
		/// <param name="Second dimention"> - position in FIRST HIDDEN layer</param>
		float first_layer	[hidden_size]		[input];

		/// <summary>
		/// Weights between hidden layers. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - number between which hidden layers (0 - between first hidden and second hidden)</param>
		/// <param name="Second dimention"> - position in NEXT HIDDEN layer</param>
		/// <param name="Third dimention"> - position in PREVIOUS HIDDEN layer</param>
		float hidden_layers	[hidden_amount - 1]	[hidden_size]	[hidden_size];

		/// <summary>
		/// Layer between last hidden layer and output layer. PAY ATTENTION TO DIMENTIONS. They were reversed for faster access.
		/// </summary>
		/// <param name="First dimention"> - position in LAST HIDDEN layer layer</param>
		/// <param name="Second dimention"> - position in OUTPUT layer</param>
		float last_layer	[hidden_size]		[output];
	} weights;

	ann();
};

template<unsigned int input, unsigned int output, unsigned int hidden_amount, unsigned int hidden_size>
inline ann<input, output, hidden_amount, hidden_size>::ann()
{
	for (unsigned int i = 0; i < input; i++)
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
	for (unsigned int i = 0; i < output; i++)
	{
		neurons.output[i] = 0.f;
	}
}
