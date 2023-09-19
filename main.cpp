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

		// �d��, �o�C�A�X, ���̓x�N�g���̏�����
		initializeWeights(inputSize, hiddenSize);
		initializeBiases(hiddenSize, inputSize);
		initializeInputs(inputSize);

		// ���ԑw�̏o�͎�����������
		hidden.resize(hiddenSize);

		//�o�͑w�̎�����������
		output.resize(inputSize);
	}

	// �d�ݍs���������
	void initializeWeights(int inputSize, int hiddenSize) {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		// ����->����
		weight.resize(hiddenSize);
		for (size_t i = 0; i < hiddenSize; ++i) {
			weight[i].resize(inputSize);
			for (size_t j = 0; j < inputSize; ++j) {  // inputSize �ɏC��
				weight[i][j] = dist(mt);
			}
		}

		// ����->�o��
		weight2.resize(inputSize);
		for (size_t i = 0; i < inputSize; ++i) {
			weight2[i].resize(hiddenSize);
			for (size_t j = 0; j < hiddenSize; ++j) {
				weight2[i][j] = dist(mt);
			}
		}

	}

	// �o�C�A�X������
	void initializeBiases(int hiddenSize, int inputSize) {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		// ����->����
		bias.resize(hiddenSize);
		for (size_t i = 0; i < hiddenSize; ++i) {
			bias[i] = dist(mt);
		}

		// ����->�o��
		bias2.resize(inputSize);
		for (size_t i = 0; i < inputSize; ++i) {
			bias2[i] = dist(mt);
		}
	}

	// ���͏�����
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

	// ���́`�B��w����sigmoid�܂ł̉��Z
	void calcHiddenLayer() {
		for (size_t i = 0; i < hidden.size(); ++i) {
			for (size_t j = 0; j < x.size(); ++j) {
				// ���͂Əd�݂̐�
				hidden[i] += x[j] * weight[i][j];  // weight[i][j] �ɏC��
			}
			// �o�C�A�X�̉��Z
			hidden[i] += bias[i];
			hidden[i] = relu(hidden[i]);
		}
	}

	// �B��w�`�o�͑w�̌v�Z
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
	// �C���X�^���X����
	SimpleNeuralNetwork nn(2, 4);

	//�v�Z�E�o��
	nn.calcHiddenLayer();
	nn.calcOutputLayer();
	//nn.printHiddenLayer();
	
	// ����
	std::cout << "input:";
	for (double value : nn.x) {
		std::cout << value << " ";
	}
	std::cout << std::endl;

	// �d��
	std::cout << "weight: \n";
	for (size_t i = 0; i < nn.weight.size(); ++i) {
		for (double value : nn.weight[i]) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}

	// �o�C�A�X
	std::cout << "bias: ";
	for (double value : nn.bias) {
		std::cout << value << " ";
	}
	std::cout << std::endl;

	// �B��w
	std::cout << "weight: ";
	for (size_t i = 0; i < nn.hidden.size(); ++i) {
		std::cout << nn.hidden[i] << " ";
	}
	std::cout << std::endl;

	// �d��2
	std::cout << "weight2:";
	for (size_t i = 0; i < nn.output.size(); ++i) {
		for (double value : nn.weight2[i]) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}

	//�o�C�A�X2
	std::cout << "bias2:";
	for (double value : nn.bias2) {
		std::cout << value << " ";
	}
	std::cout << std::endl;

	// �o��
	std::cout << "output:";
	for (size_t i = 0; i < nn.output.size(); ++i) {
		std::cout << nn.output[i] << " ";
	}
	std::cout << std::endl;


	return 0;
}
