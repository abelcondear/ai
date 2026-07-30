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
#include <array>
#include <unistd.h>
#include <sys/wait.h>


std::string exec(const char* cmd) {
     std::array<char, 128> buffer;
     std::string result;

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-attributes"
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    #pragma GCC diagnostic pop


    if (!pipe) {
             throw std::runtime_error("popen() failed.");
    }

     while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
             result += buffer.data();
     }

     return result;
}


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


CustomModel getModel(
     std::vector<std::vector<std::tuple<std::string, float, float, float, float, float, float>>> values
) {
     CustomModel model(
             43,	// 43 columns-fixed
             1,	// 1 row-fixed
             1	// 1 depth
     );

     std::vector<float> open_column;
     std::vector<float> close_column;

     float open_value = 0.0;
     float close_value = 0.0;

     std::vector<std::tuple<std::string, float, float, float, float, float, float>>::iterator g;

     torch::Tensor input_tensor;
     torch::Tensor input;

     for (int x = 0; x < values.size(); x ++) {
		g = values[x].begin();

		open_value = std::get<1>(*g);
		close_value = std::get<4>(*g);

		open_column.push_back(open_value);
		close_column.push_back(close_value);
	}



	input_tensor = torch::tensor(
		{
		open_column[0],
		open_column[1],
		open_column[2],
		open_column[3],
		open_column[4],
		open_column[5],
		open_column[6],
		open_column[7],
		open_column[8],
		open_column[9],
		open_column[10],
		open_column[11],
		open_column[12],
		open_column[13],
		open_column[14],
		open_column[15],
		open_column[16],
		open_column[17],
		open_column[18],
		open_column[19],
		open_column[20],
		open_column[21],
		open_column[22],
		open_column[23],
		open_column[24],
		open_column[25],
		open_column[26],
		open_column[27],
		open_column[28],
		open_column[29],
		open_column[30],
		open_column[31],
		open_column[32],
		open_column[33],
		open_column[34],
		open_column[35],
		open_column[36],
		open_column[37],
		open_column[38],
		open_column[39],
		open_column[40],
		open_column[41],
		open_column[42]
		},
		torch::kFloat32
	);

     std::cout << "//--------------------------" << std::endl;
     std::cout << "// open column" << std::endl;
     std::cout << "//--------------------------" << std::endl;

     std::cout << "//--------------------------" << std::endl;
     std::cout << input_tensor << std::endl;
     std::cout << "//--------------------------" << std::endl;

     std::cout << std::endl;

     input = model.forward(input_tensor);

     std::cout << "//--------------------------" << std::endl;
     std::cout << input << std::endl;
     std::cout << "//--------------------------" << std::endl;



	input_tensor = torch::tensor(
     	{
		close_column[0],
		close_column[1],
		close_column[2],
		close_column[3],
		close_column[4],
		close_column[5],
		close_column[6],
		close_column[7],
		close_column[8],
		close_column[9],
		close_column[10],
		close_column[11],
		close_column[12],
		close_column[13],
		close_column[14],
		close_column[15],
		close_column[16],
		close_column[17],
		close_column[18],
		close_column[19],
		close_column[20],
		close_column[21],
		close_column[22],
		close_column[23],
		close_column[24],
		close_column[25],
		close_column[26],
		close_column[27],
		close_column[28],
		close_column[29],
		close_column[30],
		close_column[31],
		close_column[32],
		close_column[33],
		close_column[34],
		close_column[35],
		close_column[36],
		close_column[37],
		close_column[38],
		close_column[39],
		close_column[40],
		close_column[41],
		close_column[42]
		},
     	torch::kFloat32
);

     std::cout << "//--------------------------" << std::endl;
     std::cout << "// close column " << std::endl;
     std::cout << "//--------------------------" << std::endl;

     std::cout << "//--------------------------" << std::endl;
     std::cout << input_tensor << std::endl;
     std::cout << "//--------------------------" << std::endl;

     std::cout << std::endl;

     input = model.forward(input_tensor);

     std::cout << "//--------------------------" << std::endl;
     std::cout << input << std::endl;
     std::cout << "//--------------------------" << std::endl;


     std::cout << std::endl;
     std::cout << std::endl;

     return model;
}


CustomModel readCSV
(
     const std::string& filePath,
     const char chrType,
     const short int lnType,
     const short int column
) {
     CustomModel model(1, 1, 1);

     int numLines = 0;

     std::ifstream in(filePath);
     std::string unused;

     while ( std::getline(in, unused) )
             ++ numLines;

     -- numLines;

     std::ifstream file(filePath);

     if (!file.is_open()) {
             std::cerr << "Error: Could not open the file." << std::endl;
             return model;
     }

     std::string line;

     std::vector<std::vector<std::string>> rows;

     int config_rows = numLines;

     int config_cols = 6;

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

     std::string date_str;
     float value_ft_open;
     float value_ft_high;
     float value_ft_low;
     float value_ft_close;
     float value_ft_adj_close;
     float value_ft_volume;

     int rowline = 0;

     for (const auto& row : rows) {
             rowline ++;

             if (row.size()) {
                     try {
                             std::cout << "line: " << rowline << std::endl;

                             if (rowline == numLines + 1) { break; }

                             auto it = (*itt).begin();

                             itt ++;

                             it = (*itt).begin();

                             date_str = *it;
                             ++ it;

                             value_ft_open = std::stof(*it);
                             ++ it;

                             value_ft_high = std::stof(*it);
                             ++ it;

                             value_ft_low = std::stof(*it);
                             ++ it;

                             value_ft_close = std::stof(*it);
                             ++ it;

                             value_ft_adj_close = std::stof(*it);
                             ++ it;

                             value_ft_volume = std::stof(*it);

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

                             rowArr.erase(
                                     rowArr.begin(),
                                     rowArr.end()
                             );

                             std::cout << std::endl;
                     }
                     catch (const std::exception& e) {
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

     short int column = paramColumn;

     CustomModel model = readCSV(filePath, chrType, lnType, column);

     return 0;
}


