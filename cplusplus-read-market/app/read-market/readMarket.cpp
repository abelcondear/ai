#include <stdio.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>

// ---------------------------
// linux terminal command
// ---------------------------
//  apt install python3-torch
//  cmake --build . --config Release
//  cmake -S . -B ./build/
//  cmake -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
//  cp readMarket.cpp ./build/ && cmake --build ./build --config Release && ./build/readMarket
//  cp readMarket.cpp ./build/ && cmake --build ./build --config Release && ./build/readMarket > output.log && nano output.log
// ---------------------------

enum lineType {
	all		= 0,
	individual	= 1
};

enum splitType {
	blank		= ' ',
	tab		= '\t',
	comma		= ',',
	semicolon	= ';'
};

struct CustomModel : torch::nn::Module {
	torch::nn::Linear fc1{nullptr};
	torch::nn::Linear fc2{nullptr};

	CustomModel(
		int64_t input_size,
		int64_t hidden_size,
		int64_t output_size
    	) {
        	fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        	fc2 = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
    	}

    	torch::Tensor forward(torch::Tensor x) {
		//convert all tensor elements to float type by default
        	x = x.to(torch::kFloat32);

        	//convert it into one dimension before executing forward method
        	x = x.reshape(x.sizes());

        	x = torch::relu(fc1->forward(x));
        	x = fc2->forward(x);

        	return x;
    	}
};

//CustomModel getModel(float values[5][3]) {
//parameter values (data type updated)
CustomModel getModel(std::vector<std::vector<float>> values) {
    CustomModel module(values[0].size(), values.size(), 1);

    // CustomModel ----
    //  values[0].size()=columns
    //  values.size()=rows
    //  1=depth
    // ---------------

    torch::Tensor input_tensor;
    torch::Tensor input;

    std::cout << std::endl;
    std::cout << std::endl;

    for (int x = 0; x < values.size(); x ++) {
    	input_tensor = torch::tensor(
        	{
			values[x][0],
			values[x][1],
			values[x][2],
			values[x][3],
			values[x][4],
			values[x][5]
		},
        	torch::kFloat32
    	);

	std::cout << "--------------------------" << std::endl;
	std::cout << "values [row=" << (x + 1) << "]:" << std::endl;
	std::cout << std::endl;
	std::cout << values[x][0] << std::endl;
	std::cout << values[x][1] << std::endl;
	std::cout << values[x][2] << std::endl;
	std::cout << values[x][3] << std::endl;
	std::cout << values[x][4] << std::endl;
	std::cout << values[x][5] << std::endl;
	std::cout << std::endl;
	std::cout << "--------------------------" << std::endl;

	std::cout << std::endl;

	std::cout << "--------------------------" << std::endl;
	std::cout << "input_tensor:" << std::endl;
	std::cout << input_tensor << std::endl;
	std::cout << "--------------------------" << std::endl;

	std::cout << std::endl;

	input = module.forward(input_tensor);

	std::cout << "--------------------------" << std::endl;
	std::cout << "input [after forward method calling]:" << std::endl;
	std::cout << input << std::endl;
	std::cout << "--------------------------" << std::endl;

	std::cout << std::endl;
    }

    return module;
}

const char* applyFunction(auto& value) {
	// predict next value of a serie of values
	std::cout << "calling applyFunction ..." << std::endl;
	return ""; //empty string (result) for the moment
}

CustomModel readCSV
(
	const std::string& filePath,
	const char chrType,
	const short int lnType,
	const short int column
) {
	CustomModel model(1, 1, 1); //create model using initial values by default

	int numLines = 0;
	std::ifstream in(filePath);
	std::string unused;

	while ( std::getline(in, unused) )
   		++ numLines;

	-- numLines; // do not count headers (first line)

	std::ifstream file(filePath);

	if (!file.is_open()) {
		std::cerr << "Error: Could not open the file!" << std::endl;
		return model; //empty model
	}

	std::string line;

	std::vector<std::vector<std::string>> rows;

	//these values should be set according to csv file
	//according to csv file (dynamically setted)
    	int config_rows = numLines;

	//according to csv file (statically setted)
	//(could be change to dynamical setting later)
	//(columns with numeric value)
	int config_cols = 6;

	//change values (fixed array) into values (dynamic array)
	std::vector<std::vector<float>> values(config_rows, std::vector<float>(config_cols, 0.0f));

	while (std::getline(file, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<std::string> row;

		while
		(
			std::getline(lineStream, cell, chrType)
		)
		{
			row.push_back(cell);
		}

		rows.push_back(row);
	}

	std::cout
		<< "Print each value from every line of csv file."
		<< std::endl
		<< std::endl;

	int numline = 0;
	int numcol = 0;

	short int n_col = 0;
	short int n_row = 0;

	for (const auto& row : rows) {
		if (row.size()) {
			std::cout << "line: " << numline << std::endl;

			numline ++;

			if (numline == 1) { continue; } //jump headers from csv file - continue next iteration

			n_row ++;

			numcol = 0;
			n_col = 0;

			for (const auto& value : row) {
				numcol ++;

				if (numcol == 1) { continue; } //jump date column from csv file - continue next iteration

				n_col ++;

				float value_f = std::stof(value);

				std::cout
					<< "-------------------"
					<< std::endl;

				std::cout
					<< "value:"
					<< "["
					<< value_f
					<< "]"
					<< std::endl;

				values[ n_row - 1 ][ n_col - 1 ] = value_f;

				std::cout
					<< "-------------------"
					<< std::endl;

				//if (lnType == lineType::individual && n_col == column) {
				//	std::cout
				//		<< "applying function for \""
				//		<< value
				//		<< "\" ..."
				//		<< std::endl;
				//
				//	applyFunction(value);
				//}
			}

			std::cout << std::endl;
		}
	}

	model = getModel(values);

	file.close();

	return model;
}

int main(int argc, char *argv[]) {
	//testing using real file history from Yahoo Finance 
	const std::string filePath = "./apple-inc-appl.csv";
	short int column;
	lineType lnType;
	splitType chrType;

	//./build/readMarket --line-type "individual" --column-number 2

	//char* paramLineType = argv[2];
	//int paramColumn = atoi(argv[4]);

	//std::cout << "Reading parameters ..." << std::endl;

	//std::cout << std::endl;

	//std::cout << "line-type =" << paramLineType << std::endl;
	//std::cout << "column-number=" << paramColumn << std::endl;

	//std::cout << std::endl;

	//std::cout << "End reading parameters ..." << std::endl;
	//std::cout << std::endl;

	//if (strcmp(paramLineType, "all") == 0) { lnType = lineType::all;  }
	//if (strcmp(paramLineType, "individual") == 0) { lnType = lineType::individual; }

	chrType = splitType::semicolon;
	lnType = lineType::individual;
	//column = paramColumn;
	column = 1;

	CustomModel model = readCSV(filePath, chrType, lnType, column);

	return 0;
}
