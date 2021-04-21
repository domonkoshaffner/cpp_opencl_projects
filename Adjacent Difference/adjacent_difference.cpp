#include <CL/cl2.hpp>

#include <chrono>
#include <numeric>
#include <iterator>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE


int main()
{
    try
    {
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();

        // Load program source
        std::ifstream source_file{ "C:/Users/haffn/Desktop/MSc-III/GPU-II/Projects/adjacent_difference/adjacent_difference.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./adjacent_difference.cl" };

        // Create program and kernel
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file }, std::istreambuf_iterator<char>{} } };
        program.build({ device });

        auto adjacent_difference = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "adjacent_difference");

        // Init computation
        const std::size_t chainlength = 100000000u;
        std::vector<cl_float> vec_x(chainlength);
        std::vector<cl_float> vec_y(chainlength, 0);

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{}, distribution = std::uniform_real_distribution<cl_float>{ -100.0, 100.0 }]() mutable { return distribution(engine); };
        std::generate_n(std::begin(vec_x), chainlength, prng);

        // Creating buffers from the vectors
        cl::Buffer buf_x{std::begin(vec_x), std::end(vec_x), false};
        cl::Buffer buf_y{std::begin(vec_y), std::end(vec_y), false};


        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec_x), std::end(vec_x), buf_x);
        cl::copy(queue, std::begin(vec_y), std::end(vec_y), buf_y);

        // Starting the clock
        auto time_gpu0 = std::chrono::high_resolution_clock::now();
        // Launch kernels
        adjacent_difference(cl::EnqueueArgs{queue, cl::NDRange{ chainlength } }, buf_x, buf_y);

        cl::finish();

        // (Blocking) fetch of results
        cl::copy(queue, buf_x, std::begin(vec_x), std::end(vec_x));
        cl::copy(queue, buf_y, std::begin(vec_y), std::end(vec_y));

        // Stopping the clock and calculating the ellapsed time 
        auto time_gpu1 = std::chrono::high_resolution_clock::now();
        auto time_difference_gpu = time_gpu1 - time_gpu0;       


// ############################################################################

        //Creating the same program and running it on the CPU
        auto time_cpu0 = std::chrono::high_resolution_clock::now();
        std::adjacent_difference(vec_x.begin(), vec_x.end(), vec_x.begin());

        auto time_cpu1 = std::chrono::high_resolution_clock::now();
        auto time_difference_1 = time_cpu1 - time_cpu0;

        std::cout << std::endl << "The computational time for a " << chainlength << " element long vector on the CPU: " << time_difference_1.count()/1000000  << " milisec.";
        std::cout << std::endl << "The computational time for a " << chainlength << " element long vector on the GPU: " << time_difference_gpu.count()/1000000  << " milisec.";

// ############################################################################


        // Checking the vectors if they are equal
        std::cout << std::endl << std::endl << "Validation: ";

        if (vec_x == vec_y) {
		std::cout << "The vectors are equal" << std::endl;
	    } 
        else {
		std::cout << "The vectors are not equal" << std::endl;
	    }



    }
    catch (cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

        std::exit(error.err());
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
