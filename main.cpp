#include <iostream>
#include <vector>
#include <random>

//simpleNNClass
class SimpleNeuralNetwork {
public:
	std::vector<std::vector<double>> weight;
	std::vector<std::vector<double>> weight2;
	std::vector<double> bias;
	std::vector<double> bias2;
	std::vector<double> x;
	std::vector<double> hidden;
	std::vector<double> output;

public:
	SimpleNeuralNetwork(int inputSize, int hiddenSize) {

		// 重み, バイアス, 入力ベクトルの初期化
		initializeWeights(inputSize, hiddenSize);
		initializeBiases(hiddenSize, inputSize);
		initializeInputs(inputSize);

		// 中間層の出力次元を初期化
		hidden.resize(hiddenSize);

		//出力層の次元を初期化
		output.resize(inputSize);
	}

	// 重み行列を初期化
	void initializeWeights(int inputSize, int hiddenSize) {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		// 入力->中間
		weight.resize(hiddenSize);
		for (size_t i = 0; i < hiddenSize; ++i) {
			weight[i].resize(inputSize);
			for (size_t j = 0; j < inputSize; ++j) {  // inputSize に修正
				weight[i][j] = dist(mt);
			}
		}

		// 中間->出力
		weight2.resize(inputSize);
		for (size_t i = 0; i < inputSize; ++i) {
			weight2[i].resize(hiddenSize);
			for (size_t j = 0; j < hiddenSize; ++j) {
				weight2[i][j] = dist(mt);
			}
		}

	}

	// バイアス初期化
	void initializeBiases(int hiddenSize, int inputSize) {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		// 入力->中間
		bias.resize(hiddenSize);
		for (size_t i = 0; i < hiddenSize; ++i) {
			bias[i] = dist(mt);
		}

		// 中間->出力
		bias2.resize(inputSize);
		for (size_t i = 0; i < inputSize; ++i) {
			bias2[i] = dist(mt);
		}
	}

	// 入力初期化
	void initializeInputs(int inputSize) {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		x.resize(inputSize);
		for (size_t i = 0; i < inputSize; ++i) {
			x[i] = dist(mt);
		}
	}

	// relu
	double relu(double x) {
		return std::max(0.0, x);
	}

	// 入力～隠れ層からsigmoidまでの演算
	void calcHiddenLayer() {
		for (size_t i = 0; i < hidden.size(); ++i) {
			for (size_t j = 0; j < x.size(); ++j) {
				// 入力と重みの積
				hidden[i] += x[j] * weight[i][j];  // weight[i][j] に修正
			}
			// バイアスの加算
			hidden[i] += bias[i];
			hidden[i] = relu(hidden[i]);
		}
	}

	// 隠れ層～出力層の計算
	void calcOutputLayer() {
		for (size_t i = 0; i < output.size(); ++i) {
			for (size_t j = 0; j < hidden.size(); ++j) {
				output[i] += hidden[j] * weight2[i][j];
			}
			output[i] += bias2[i];
		}
	}

};

int main() {
	// インスタンス生成
	SimpleNeuralNetwork nn(2, 4);

	//計算・出力
	nn.calcHiddenLayer();
	nn.calcOutputLayer();
	//nn.printHiddenLayer();
	
	// 入力
	std::cout << "input:";
	for (double value : nn.x) {
		std::cout << value << " ";
	}
	std::cout << std::endl;

	// 重み
	std::cout << "weight: \n";
	for (size_t i = 0; i < nn.weight.size(); ++i) {
		for (double value : nn.weight[i]) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}

	// バイアス
	std::cout << "bias: ";
	for (double value : nn.bias) {
		std::cout << value << " ";
	}
	std::cout << std::endl;

	// 隠れ層
	std::cout << "weight: ";
	for (size_t i = 0; i < nn.hidden.size(); ++i) {
		std::cout << nn.hidden[i] << " ";
	}
	std::cout << std::endl;

	// 重み2
	std::cout << "weight2:";
	for (size_t i = 0; i < nn.output.size(); ++i) {
		for (double value : nn.weight2[i]) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}

	//バイアス2
	std::cout << "bias2:";
	for (double value : nn.bias2) {
		std::cout << value << " ";
	}
	std::cout << std::endl;

	// 出力
	std::cout << "output:";
	for (size_t i = 0; i < nn.output.size(); ++i) {
		std::cout << nn.output[i] << " ";
	}
	std::cout << std::endl;


	return 0;
}
