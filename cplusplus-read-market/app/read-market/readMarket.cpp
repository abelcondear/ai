#include <stdio.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <string>
#include <cstring>
#include <memory>
#include <typeinfo>
#include <cxxabi.h>
#include <stdexcept>

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
		//convert all tensor elements
		//to float type by default
        	x = x.to(torch::kFloat32);

        	//convert it into one dimension
		//before executing forward method
        	x = x.reshape(x.sizes());

        	x = torch::relu(fc1->forward(x));
        	x = fc2->forward(x);

        	return x;
    	}
};

CustomModel getModel(std::vector<std::vector<std::tuple<std::string, float, float, float, float, float, float>>> values) {
	CustomModel model(
		6, // 6 columns-fixed
		values.size(),
		1
	);

	// CustomModel ----
	//  3=columns
	//  5=rows
    	//  1=depth
    	// ---------------

    	torch::Tensor input_tensor;
    	torch::Tensor input;

	for (int x = 0; x < values.size(); x ++) { // read elements from values parameter
    	//for (int x = 0; x < values.size(); x ++) {

		std::vector<std::tuple<std::string, float, float, float, float, float, float>>::iterator g = values[x].begin(); // get vector item

		std::string date_str = std::get<0>(*g); // get tuple item
		float open_ft = std::get<1>(*g); // get tuple item
		float high_ft = std::get<2>(*g); // get tuple item
		float low_ft = std::get<3>(*g); // get tuple item
		float close_ft = std::get<4>(*g); // get tuple item
		float adj_ft = std::get<5>(*g); // get tuple item
		float volumne_ft = std::get<6>(*g); // get tuple item

		// then, onvert one column in a single parameters list
        	input_tensor = torch::tensor(
                	{
				open_ft,
				high_ft,
				low_ft,
				close_ft,
				adj_ft,
				volumne_ft
                	},
                	torch::kFloat32
        	);

	        std::cout << "--------------------------" << std::endl;
        	std::cout << "input_tensor:" << std::endl;
	        std::cout << input_tensor << std::endl;
        	std::cout << "--------------------------" << std::endl;

	        std::cout << std::endl;

        	input = model.forward(input_tensor);

	        std::cout << "--------------------------" << std::endl;
        	std::cout << "input [after forward method calling]:" << std::endl;
	        std::cout << input << std::endl;
        	std::cout << "--------------------------" << std::endl;

        	std::cout << std::endl;

    	}

	return model;
}

CustomModel readCSV
(
	const std::string& filePath,
	const char chrType,
	const short int lnType,
	const short int column
) {
	//create a dummy model
	CustomModel model(1, 1, 1);

	int numLines = 0;

	std::ifstream in(filePath);
	std::string unused;

	while ( std::getline(in, unused) )
   		++ numLines;

	// do not count headers (first line)
	-- numLines;

	std::ifstream file(filePath);

	if (!file.is_open()) {
		std::cerr << "Error: Could not open the file!" << std::endl;
		return model; //return empty model
	}

	std::string line;

	std::vector<std::vector<std::string>> rows;

	//these values should be set according to csv file
	//according to csv file (dynamically setted)
    	int config_rows = numLines;

	//according to csv file (statically setted)
	//(configuration amount columns for numeric values)
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

	std::vector<std::vector<std::tuple<std::string, float, float, float, float, float, float>>> myArr;
	std::vector<std::tuple<std::string, float, float, float, float, float, float>> rowArr;

	std::vector< std::vector< std::__cxx11::basic_string<char> > >::iterator itt_begin = rows.begin();
	std::vector< std::vector< std::__cxx11::basic_string<char> > >::iterator itt_end = rows.end();
	std::vector< std::vector< std::__cxx11::basic_string<char> > >::iterator itt = itt_begin;

	// get data type from auto variable type
	std::string date_str;		// date column
	float value_ft_open;		// open column
	float value_ft_high;		// high column
	float value_ft_low;		// low column
	float value_ft_close;		// close column
	float value_ft_adj_close;	// adj. close column
	float value_ft_volume;		// volumn column

	int rowline = 0;

	for (const auto& row : rows) {
		rowline ++;

		if (row.size()) {
			try {
				//std::cout << "line-***: " << rowline << std::endl;

				if (rowline == numLines + 1) { break; }
				// -------------------------------------

				// header - csv file
				// Date;Open;High;Low;Close;Adj-Close;Volume;

				// -------------------------------------

				//place on first element from vector
    				auto it = (*itt).begin();

				itt ++;

				it = (*itt).begin();

				date_str = *it;
				//std::cout << date_str << std::endl;
				++ it;

				value_ft_open = std::stof(*it);
				//std::cout << value_ft_open << std::endl;
	    			++ it;

				value_ft_high = std::stof(*it);
				//std::cout << value_ft_high << std::endl;
	    			++ it;

				value_ft_low = std::stof(*it);
				//std::cout << value_ft_low << std::endl;
	    			++ it;

				value_ft_close = std::stof(*it);
				//std::cout << value_ft_close << std::endl;
	    			++ it;

				value_ft_adj_close = std::stof(*it);
				//std::cout << value_ft_adj_close << std::endl;
    				++ it;

				value_ft_volume = std::stof(*it);
				//std::cout << value_ft_volume << std::endl;

				//inserting columns to single row
				rowArr.push_back(
					std::make_tuple(
						date_str,
						value_ft_open,
						value_ft_high,
						value_ft_low,
						value_ft_close,
						value_ft_adj_close,
						value_ft_volume
					)
				);

				myArr.push_back(rowArr);

				//clean up before inserting new element
				//empty rowArr (row of myArr)
				rowArr.erase(
					rowArr.begin(),
					rowArr.end()
				);

				std::cout << std::endl;
			}
			catch (const std::exception& e) {
				//break; // stop for loop
				continue;
			}

		}
	}

	std::cout << std::endl;
	std::cout << std::endl;

	file.close();

	return getModel(myArr);
}

int main(int argc, char *argv[]) {
	const std::string filePath = "./apple-inc-appl.csv";

	short int lnType = 1;
	char chrType = ';';
	int paramColumn = 1;

	std::cout << std::endl;
	std::cout << std::endl;

	short int column = paramColumn;

	CustomModel model = readCSV(filePath, chrType, lnType, column);

	return 0;
}
