#include <fstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cstring>

int main(int argc, char** argv) {
    std::ofstream outputFile("readMarket.cpp");

    char slash = 47; // "/" slash
    char quote = 34; // '"' quote

    int amount_tensor = std::stoi(argv[1]);

    // ----------------------------------------

    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not create or open the file." << std::endl;
        return 1;
    }

    // ----------------------------------------

    outputFile << "#include <stdio.h>" << std::endl;
    outputFile << "#include <torch/torch.h>" << std::endl;
    outputFile << "#include <torch/script.h>" << std::endl;
    outputFile << "#include <iostream>" << std::endl;
    outputFile << "#include <fstream>" << std::endl;
    outputFile << "#include <sstream>" << std::endl;
    outputFile << "#include <vector>" << std::endl;
    outputFile << "#include <tuple>" << std::endl;
    outputFile << "#include <string>" << std::endl;
    outputFile << "#include <cstring>" << std::endl;
    outputFile << "#include <memory>" << std::endl;
    outputFile << "#include <typeinfo>" << std::endl;
    outputFile << "#include <cxxabi.h>" << std::endl;
    outputFile << "#include <stdexcept>" << std::endl;
    outputFile << "#include <array>" << std::endl;
    outputFile << "#include <unistd.h>" << std::endl;
    outputFile << "#include <sys/wait.h>" << std::endl;

    outputFile << std::endl;
    outputFile << std::endl;

    // ----------------------------------------

    outputFile << "std::string exec(const char* cmd) {" << std::endl;
    outputFile << "     std::array<char, 128> buffer;" << std::endl;
    outputFile << "     std::string result;" << std::endl;
    outputFile << std::endl;
    outputFile << "    #pragma GCC diagnostic push" << std::endl;
    outputFile << "    #pragma GCC diagnostic ignored \"-Wignored-attributes\"" << std::endl;
    outputFile << "    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, \"r\"), pclose);" << std::endl;
    outputFile << "    #pragma GCC diagnostic pop" << std::endl;
    outputFile << std::endl;
    outputFile << std::endl;
    outputFile << "    if (!pipe) {" << std::endl;
    outputFile << "             throw std::runtime_error(\"popen() failed.\");" << std::endl;
    outputFile << "    }" << std::endl;
    outputFile << std::endl;
    outputFile << "     while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {" << std::endl;
    outputFile << "             result += buffer.data();" << std::endl;
    outputFile << "     }" << std::endl;
    outputFile << std::endl;
    outputFile << "     return result;" << std::endl;
    outputFile << "}" << std::endl;

    // --------------------------------------------------

    outputFile << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------

    outputFile << "struct CustomModel : torch::nn::Module {" << std::endl;
    outputFile << "     torch::nn::Linear fc1{nullptr};" << std::endl;
    outputFile << "     torch::nn::Linear fc2{nullptr};" << std::endl;
    outputFile << std::endl;
    outputFile << "     CustomModel(" << std::endl;
    outputFile << "             int64_t input_size," << std::endl;
    outputFile << "             int64_t hidden_size," << std::endl;
    outputFile << "             int64_t output_size" << std::endl;
    outputFile << "     ) {" << std::endl;
    outputFile << "             fc1 = register_module(\"fc1\", torch::nn::Linear(input_size, hidden_size));" << std::endl;
    outputFile << "             fc2 = register_module(\"fc2\", torch::nn::Linear(hidden_size, output_size));" << std::endl;
    outputFile << "     }" << std::endl;
    outputFile << std::endl;
    outputFile << "     torch::Tensor forward(torch::Tensor x) {" << std::endl;
    outputFile << "             x = x.to(torch::kFloat32);" << std::endl;
    outputFile << std::endl;
    outputFile << "             x = x.reshape(x.sizes());" << std::endl;
    outputFile << std::endl;
    outputFile << "             x = torch::relu(fc1->forward(x));" << std::endl;
    outputFile << "             x = fc2->forward(x);" << std::endl;
    outputFile << std::endl;
    outputFile << "             return x;" << std::endl;
    outputFile << "     }" << std::endl;
    outputFile << "};" << std::endl;

    // --------------------------------------------------

    outputFile << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------
    // CustomModel :: getModel  - BEGIN

    outputFile << "CustomModel getModel(" << std::endl;
    outputFile << "     std::vector<std::vector<std::tuple<std::string, float, float, float, float, float, float>>> values" << std::endl;
    outputFile << ") {" << std::endl;
    outputFile << "     CustomModel model(" << std::endl;
    outputFile << "             43, " << slash << slash << " 43 columns-fixed" << std::endl;
    outputFile << "             1, " << slash << slash << " 1 row-fixed" << std::endl;
    outputFile << "             1 " << slash << slash << " 1 depth" << std::endl;
    outputFile << "     );" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::vector<float> open_column;" << std::endl;
    outputFile << "     std::vector<float> close_column;" << std::endl;
    outputFile << std::endl;
    outputFile << "     float open_value = 0.0;" << std::endl;
    outputFile << "     float close_value = 0.0;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::vector<std::tuple<std::string, float, float, float, float, float, float>>::iterator g;" << std::endl;
    outputFile << std::endl;
    outputFile << "     torch::Tensor input_tensor;" << std::endl;
    outputFile << "     torch::Tensor input;" << std::endl;
    outputFile << std::endl;
    outputFile << "	for (int x = 0; x < values.size(); x ++) {" << std::endl;
    outputFile << "		g = values[x].begin();" << std::endl;
    outputFile << std::endl;
    outputFile << "		open_value = std::get<1>(*g);" << std::endl;
    outputFile << "		close_value = std::get<4>(*g);" << std::endl;
    outputFile << std::endl;
    outputFile << "		open_column.push_back(open_value);" << std::endl;
    outputFile << "		close_column.push_back(close_value);" << std::endl;
    outputFile << "	}" << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------

    outputFile << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------


    // --------------------------------------------------
    // --------------------------------------------------

    outputFile << "	input_tensor = torch::tensor(" << std::endl;
    outputFile << "		{" << std::endl;

    for (int index = 0; index < amount_tensor; index ++) {
        if (index + 1 == amount_tensor) {

    outputFile << "			open_column[" << index << "]" << std::endl;

        } else {

     outputFile << "			open_column[" << index << "]," << std::endl;

        }
    }

    outputFile << "		}," << std::endl;
    outputFile << "		torch::kFloat32" << std::endl;
    outputFile << "	);" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << " open column" << quote << " << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << "  << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << "     std::cout << input_tensor << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     input = model.forward(input_tensor);" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << " << quote  << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << "     std::cout << input << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------
    // --------------------------------------------------


    // --------------------------------------------------

    outputFile << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------


    // --------------------------------------------------
    // --------------------------------------------------

    outputFile << "	input_tensor = torch::tensor(" << std::endl;
    outputFile << "     	{" << std::endl;

    for (int index = 0; index < amount_tensor; index ++) {
        if ((index + 1) == amount_tensor) {

    outputFile << "			close_column[" << index << "]" << std::endl;

        } else {

    outputFile << "			close_column[" << index << "]," << std::endl;

        }
    }

    outputFile << "		}," << std::endl;
    outputFile << "     	torch::kFloat32" << std::endl;
    outputFile << "	);" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << " close column \" << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << "     std::cout << input_tensor << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     input = model.forward(input_tensor);" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << "     std::cout << input << std::endl;" << std::endl;
    outputFile << "     std::cout << " << quote << slash << slash << "--------------------------" << quote << " << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << std::endl;" << std::endl;
    outputFile << "     std::cout << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     return model;" << std::endl;
    outputFile << "}" << std::endl;

    // CustomModel :: getModel  - END
    // --------------------------------------------------


    // --------------------------------------------------

    outputFile << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------


    // --------------------------------------------------

    outputFile << "CustomModel readCSV" << std::endl;
    outputFile << "(" << std::endl;
    outputFile << "     const std::string& filePath," << std::endl;
    outputFile << "     const char chrType," << std::endl;
    outputFile << "     const short int lnType," << std::endl;
    outputFile << "     const short int column" << std::endl;
    outputFile << ") {" << std::endl;
    outputFile << "     CustomModel model(1, 1, 1);" << std::endl;
    outputFile << std::endl;
    outputFile << "     int numLines = 0;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::ifstream in(filePath);" << std::endl;
    outputFile << "     std::string unused;" << std::endl;
    outputFile << std::endl;
    outputFile << "     while ( std::getline(in, unused) )" << std::endl;
    outputFile << "             ++ numLines;" << std::endl;
    outputFile << std::endl;
    outputFile << "     -- numLines;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::ifstream file(filePath);" << std::endl;
    outputFile << std::endl;
    outputFile << "     if (!file.is_open()) {" << std::endl;
    outputFile << "             std::cerr << \"Error: Could not open the file.\" << std::endl;" << std::endl;
    outputFile << "             return model;" << std::endl;
    outputFile << "     }" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::string line;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::vector<std::vector<std::string>> rows;" << std::endl;
    outputFile << std::endl;
    outputFile << "     int config_rows = numLines;" << std::endl;
    outputFile << std::endl;
    outputFile << "     int config_cols = 6;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::vector<std::vector<float>> values(config_rows, std::vector<float>(config_cols, 0.0f));" << std::endl;
    outputFile << std::endl;
    outputFile << "     while (std::getline(file, line)) {" << std::endl;
    outputFile << "             std::stringstream lineStream(line);" << std::endl;
    outputFile << "             std::string cell;" << std::endl;
    outputFile << "             std::vector<std::string> row;" << std::endl;
    outputFile << std::endl;
    outputFile << "             while" << std::endl;
    outputFile << "             (" << std::endl;
    outputFile << "                     std::getline(lineStream, cell, chrType)" << std::endl;
    outputFile << "             )" << std::endl;
    outputFile << "             {" << std::endl;
    outputFile << "                     row.push_back(cell);" << std::endl;
    outputFile << "             }" << std::endl;
    outputFile << std::endl;
    outputFile << "             rows.push_back(row);" << std::endl;
    outputFile << "     }" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::vector<std::vector<std::tuple<std::string, float, float, float, float, float, float>>> myArr;" << std::endl;
    outputFile << "     std::vector<std::tuple<std::string, float, float, float, float, float, float>> rowArr;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::vector< std::vector< std::__cxx11::basic_string<char> > >::iterator itt_begin = rows.begin();" << std::endl;
    outputFile << "     std::vector< std::vector< std::__cxx11::basic_string<char> > >::iterator itt_end = rows.end();" << std::endl;
    outputFile << "     std::vector< std::vector< std::__cxx11::basic_string<char> > >::iterator itt = itt_begin;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::string date_str;" << std::endl;
    outputFile << "     float value_ft_open;" << std::endl;
    outputFile << "     float value_ft_high;" << std::endl;
    outputFile << "     float value_ft_low;" << std::endl;
    outputFile << "     float value_ft_close;" << std::endl;
    outputFile << "     float value_ft_adj_close;" << std::endl;
    outputFile << "     float value_ft_volume;" << std::endl;
    outputFile << std::endl;
    outputFile << "     int rowline = 0;" << std::endl;
    outputFile << std::endl;
    outputFile << "     for (const auto& row : rows) {" << std::endl;
    outputFile << "             rowline ++;" << std::endl;
    outputFile << std::endl;
    outputFile << "             if (row.size()) {" << std::endl;
    outputFile << "                     try {" << std::endl;
    outputFile << "                             " << slash << slash << "std::cout << \"line: \" << rowline << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             if (rowline == numLines + 1) { break; }" << std::endl;
    outputFile << std::endl;
    outputFile << "                             auto it = (*itt).begin();" << std::endl;
    outputFile << std::endl;
    outputFile << "                             itt ++;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             it = (*itt).begin();" << std::endl;
    outputFile << std::endl;
    outputFile << "                             date_str = *it;" << std::endl;
    outputFile << "                             ++ it;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             value_ft_open = std::stof(*it);" << std::endl;
    outputFile << "                             ++ it;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             value_ft_high = std::stof(*it);" << std::endl;
    outputFile << "                             ++ it;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             value_ft_low = std::stof(*it);" << std::endl;
    outputFile << "                             ++ it;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             value_ft_close = std::stof(*it);" << std::endl;
    outputFile << "                             ++ it;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             value_ft_adj_close = std::stof(*it);" << std::endl;
    outputFile << "                             ++ it;" << std::endl;
    outputFile << std::endl;
    outputFile << "                             value_ft_volume = std::stof(*it);" << std::endl;
    outputFile << std::endl;
    outputFile << "                             rowArr.push_back(" << std::endl;
    outputFile << "                                     std::make_tuple(" << std::endl;
    outputFile << "                                             date_str," << std::endl;
    outputFile << "                                             value_ft_open," << std::endl;
    outputFile << "                                             value_ft_high," << std::endl;
    outputFile << "                                             value_ft_low," << std::endl;
    outputFile << "                                             value_ft_close," << std::endl;
    outputFile << "                                             value_ft_adj_close," << std::endl;
    outputFile << "                                             value_ft_volume" << std::endl;
    outputFile << "                                     )" << std::endl;
    outputFile << "                             );" << std::endl;
    outputFile << std::endl;
    outputFile << "                             myArr.push_back(rowArr);" << std::endl;
    outputFile << std::endl;
    outputFile << "                             rowArr.erase(" << std::endl;
    outputFile << "                                     rowArr.begin()," << std::endl;
    outputFile << "                                     rowArr.end()" << std::endl;
    outputFile << "                             );" << std::endl;
    outputFile << std::endl;
    outputFile << "                             " << slash << slash << "std::cout << std::endl;" << std::endl;
    outputFile << "                     }" << std::endl;
    outputFile << "                     catch (const std::exception& e) {" << std::endl;
    outputFile << "                             continue;" << std::endl;
    outputFile << "                     }" << std::endl;
    outputFile << std::endl;
    outputFile << "             }" << std::endl;
    outputFile << "     }" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << std::endl;" << std::endl;
    outputFile << "     std::cout << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     file.close();" << std::endl;
    outputFile << std::endl;
    outputFile << "     return getModel(myArr);" << std::endl;
    outputFile << "}" << std::endl;

    // --------------------------------------------------


    // --------------------------------------------------

    outputFile << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------


    // --------------------------------------------------

    outputFile << "int main(int argc, char *argv[]) {" << std::endl;
    outputFile << "     const std::string filePath = " << quote << "./apple-inc-appl.csv" << quote << ";" << std::endl;
    outputFile << std::endl;
    outputFile << "     short int lnType = 1;" << std::endl;
    outputFile << "     char chrType = ';';" << std::endl;
    outputFile << "     int paramColumn = 1;" << std::endl;
    outputFile << std::endl;
    outputFile << "     std::cout << std::endl;" << std::endl;
    outputFile << std::endl;
    outputFile << "     short int column = paramColumn;" << std::endl;
    outputFile << std::endl;
    outputFile << "     CustomModel model = readCSV(filePath, chrType, lnType, column);" << std::endl;
    outputFile << std::endl;
    outputFile << "     return 0;" << std::endl;
    outputFile << "}" << std::endl;

    // --------------------------------------------------

    outputFile << std::endl;
    outputFile << std::endl;

    // --------------------------------------------------


    // ----------------------------------------

    outputFile.close();

    // ----------------------------------------

    std::cout << "File generated." << std::endl;

    // ----------------------------------------

    return 0;
}
