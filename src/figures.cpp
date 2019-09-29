#include <string>
#include <thread>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pqxx/pqxx>
#include "argparse.hpp"
#include "misc_os.hpp"
#include "superpixel.hpp"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

const std::vector<std::string> images = {
        "train_images/1056.tif", // Passenger Vehicle,7
                                 // Shipping Container,12
        "train_images/1068.tif", // Fixed-wing Aircraft,1
                                 // Shipping Container,7
        "train_images/1180.tif", // Passenger Vehicle,18
                                 // Tower,32
        "train_images/1192.tif", // Passenger Vehicle,13
                                 // Shipping Container,22
                                 // Tower,3
        "train_images/2036.tif", // Passenger Vehicle,37
                                 // Shipping Container,21
        "train_images/2293.tif", // Passenger Vehicle,21
                                 // Shipping Container,36
        "train_images/2010.tif", // Tower,9
                                 // Yacht,17
        "train_images/2560.tif", // Shipping Container,16
                                 // Yacht,46

        "train_images/1127.tif", // Fixed-wing Aircraft,145
        "train_images/1438.tif"  // Yacht,75
};

int main(int argc, char* argv[]) {

}