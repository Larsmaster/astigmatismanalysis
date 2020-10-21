/*
Karlsruhe, 29.04.2020.

This programm reads the eeg data from an .easy file and calculates the CCA. The following parameters can (and should)
be adjusted: sampling rate from the eeg device, window size of the CCA, stimulation frequency, number of harmonics
and step size (distance between time windows). The CCA values are then saved to a .csv file. When choosing the window
size, keep in size that the bigger the window, the bigger the matrices in the program, which then will take longer to
be multiplied and thus affect the execution time. Don't pick a really small time window, though. The same goes for the
number of harmonics.

Change the .easy file name in line 91 and the .csv output file name in line 132.

In order to run this program, it is necessary to download the C++ library "Eigen" (more infos here:
http://eigen.tuxfamily.org/index.php?title=Main_Page). You'll probably have to tell your compiler where you saved the
Eigen folder by using the compiler flag -I path/to/Eigen/ (https://eigen.tuxfamily.org/dox/GettingStarted.html). Also,
if the program is running too slow, try compiling it with the optimization flag -O2 or -O3.

The algorithm was based on the MATLAB function canoncorr() - to see the source code of this function, type "open
canoncorr" in the command window in MATLAB.

*drops mic*
Julia Veloso de Oliveira
*/


#define _USE_MATH_DEFINES

#include <iostream>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>

// #include <Eigen/Dense>
// #include <Eigen/QR>
// #include <Eigen/SVD>

using namespace Eigen;



// define function to split tab delimited lines from the easy files
// taken from http://stackoverflow.com/a/236803/248823
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}


int main()
{

    // construct Y matrix with reference values for cca
    float f_CCA;
    int window_size, samp_rate, harmonics, step_size;
    samp_rate = 500;
    f_CCA = 7.5;
    window_size = 4;
    harmonics = 2;
    step_size = 80;

    // get array with window_size*samp_rate values in the intervall (0, window_size)
    VectorXd t = VectorXd::LinSpaced(window_size*samp_rate, 0, window_size);
    t = 2.0 * M_PI * f_CCA * t;

    // initialize Y matrix
    // number of columns = number of sample points
    // number of rows = twice the number of harmonics, since we'll be saving both sin and cos values
    MatrixXd Y = MatrixXd::Identity(window_size*samp_rate, 2*harmonics);

    // get reference sin/cos values and its harmonics
    for( int n = 0; n < harmonics; ++n )
    {
        for( int k = 0; k < window_size*samp_rate; ++k )
        {
            // since n starts at 0, add 1 as to get the right harmonics
            Y(k, (2*n)) = std::sin(t(k) * (n + 1));
            Y(k, (2*n + 1)) = std::cos(t(k) * (n + 1));
        }
    }


    // parse easy file to vector
    std::ifstream easyfile("1.easy");

    std::string line;
    std::vector<std::string> all_rows;

    // read line
    while (std::getline(easyfile, line))
    {
        std::vector<std::string> row_values;

        // the values are delimited by tabs, so the line has to be split
        split(line, '\t', row_values);

        // append to vector with all values
        all_rows.insert(all_rows.end(), row_values.begin(), row_values.end());
     }

    easyfile.close();


    // reshape vector to matrix
    // note: easy files have 13 columns, but only 7 of them (a.k.a. 7 channels) will be needed for the CCA
    int N_samples = all_rows.size()/13;
    MatrixXd X(N_samples,7);
    // get markers as well
    VectorXd markers(N_samples);

    for( int a = 0; a < N_samples; ++a )
    {
        X(a,0) = stod(all_rows[13*a]);            // channel 1
        X(a,1) = stod(all_rows[13*a + 2]);        // channel 3
        X(a,2) = stod(all_rows[13*a + 3]);        // channel 4
        X(a,3) = stod(all_rows[13*a + 4]);        // channel 5
        X(a,4) = stod(all_rows[13*a + 5]);        // channel 6
        X(a,5) = stod(all_rows[13*a + 6]);        // channel 7
        X(a,6) = stod(all_rows[13*a + 7]);        // channel 8
        markers(a) = stod(all_rows[13*a + 11]);
    }


    // open file to which the CCA values will be saved
    std::ofstream r_file("r_values_step.csv");

    // window signal and calculate CCA
    for( int k = 0; k < N_samples - window_size*samp_rate; k += step_size )
    {

        // get window_size elements from the 7 channels
        MatrixXd X_window(window_size*samp_rate,7);
        X_window = X.block(k,0,window_size*samp_rate,7);

        // CCA starts here!!!

        // mean centering
        X_window = X_window.rowwise() - X_window.colwise().mean();
        Y = Y.rowwise() - Y.colwise().mean();

        // QR Decomposition
        HouseholderQR<MatrixXd> qr_X(X_window);
        MatrixXd Q_X = qr_X.householderQ();     // Q1 in matlab canoncorr()

        HouseholderQR<MatrixXd> qr_Y(Y);
        MatrixXd Q_Y = qr_Y.householderQ();     // Q2 in matlab canoncorr()

        // matrix dimensions: both Q_X and Q_Y are of size (window_size*samp_rate, window_size*samp_rate)
        // multiply the transpose of the first 7 cols in Q_X with the first 2*harmonics cols in Q_Y
        MatrixXd product = Q_X.block(0,0,window_size*samp_rate,7).transpose()*Q_Y.block(0,0,window_size*samp_rate,2*harmonics);

        // Singular Value Decomposition
        JacobiSVD<MatrixXd> svd(product);
        VectorXd Sigma = svd.singularValues();

        // calculate r
        int d = std::min(X_window.cols(), Y.cols());
        VectorXd r(d);
        for( int i = 0; i < d; ++i )
        {
            // "remove roundoff errs" (sic) (see canoncorr())
            r(i) = std::min( std::max(Sigma(i), 0.0), 1.0 );
        }

        // save to file
        r_file << r.norm() <<',';

    }

    r_file.close();

}