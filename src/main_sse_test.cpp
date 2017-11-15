//
// Created by Yin Rochelle on 28/09/2017.
//

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"
#include "OptimizationBackend/RawResidualJacobian.h"


int main( int argc, char** argv )
{

    dso::AccumulatorApprox acc;
    acc.initialize();

    std::ifstream infile("/Users/yinr/Desktop/J.float");

    std::string line;

    dso::RawResidualJacobian J;

    while (infile >> J.resF[0] >> J.resF[1] >> J.resF[2] >> J.resF[3] >> J.resF[4] >> J.resF[5] >> J.resF[6] >> J.resF[7]
                  >> J.Jpdxi[0][0] >> J.Jpdxi[0][1] >> J.Jpdxi[0][2] >> J.Jpdxi[0][3] >> J.Jpdxi[0][4] >> J.Jpdxi[0][5]
                  >> J.Jpdxi[1][0] >> J.Jpdxi[1][1] >> J.Jpdxi[1][2] >> J.Jpdxi[1][3] >> J.Jpdxi[1][4] >> J.Jpdxi[1][5]
                  >> J.Jpdc[0][0] >> J.Jpdc[0][1] >> J.Jpdc[0][2] >> J.Jpdc[0][3]
                  >> J.Jpdc[1][0] >> J.Jpdc[1][1] >> J.Jpdc[1][2] >> J.Jpdc[1][3]
                  >> J.Jpdd[0] >> J.Jpdd[1]
                  >> J.JIdx[0][0] >> J.JIdx[0][1] >> J.JIdx[0][2] >> J.JIdx[0][3] >> J.JIdx[0][4] >> J.JIdx[0][5] >> J.JIdx[0][6] >> J.JIdx[0][7]
                  >> J.JIdx[1][0] >> J.JIdx[1][1] >> J.JIdx[1][2] >> J.JIdx[1][3] >> J.JIdx[1][4] >> J.JIdx[1][5] >> J.JIdx[1][6] >> J.JIdx[1][7]
                  >> J.JabF[0][0] >> J.JabF[0][1] >> J.JabF[0][2] >> J.JabF[0][3] >> J.JabF[0][4] >> J.JabF[0][5] >> J.JabF[0][6] >> J.JabF[0][7]
                  >> J.JabF[1][0] >> J.JabF[1][1] >> J.JabF[1][2] >> J.JabF[1][3] >> J.JabF[1][4] >> J.JabF[1][5] >> J.JabF[1][6] >> J.JabF[1][7]
                  >> J.JIdx2(0, 0) >> J.JIdx2(0, 1) >> J.JIdx2(1, 0) >> J.JIdx2(1, 1)
                  >> J.JabJIdx(0, 0) >> J.JabJIdx(0, 1) >> J.JabJIdx(1, 0) >> J.JabJIdx(1, 1)
                  >> J.Jab2(0, 0) >> J.Jab2(0, 1) >> J.Jab2(1, 0) >> J.Jab2(1, 1)
            ) {
//        std::cout << "resF:" << std::fixed << std::setprecision(8) << std::endl << J.resF << std::endl;
//        std::cout << "JabJIdx:" << std::fixed << std::setprecision(8) << std::endl << J.JabJIdx << std::endl;

        acc.update(
                J.Jpdc[0].data(), J.Jpdxi[0].data(),
                J.Jpdc[1].data(), J.Jpdxi[1].data(),
                J.JIdx2(0,0),J.JIdx2(0,1),J.JIdx2(1,1));

    }
    acc.finish();

    std::cout << std::fixed << std::setprecision(8) << std::endl << acc.H << std::endl;


    std::ofstream myfile;
    myfile.open("/Users/yinr/Desktop/sse_cpp.txt");
    for(auto r = 0; r < 13; r++) {
        for(auto c = 0; c < 13; c++) {
            myfile << std::fixed << std::setprecision(8) << acc.H(r, c) << " ";
        }
        myfile << std::endl;
    }
    myfile << std::endl;
    myfile.close();


    return 0;
}