#include <string>
#include <thread>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pqxx/pqxx>
#include "argparse.hpp"
#include "misc_os.hpp"
#include "misc_ocv.hpp"
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

void process_tif(const fs::path &dataset, const std::string &fname, const float chip_overlap, const int sp_size, bool verbose = false) {
    cv::Mat frame_raw = cv::imread(fname, cv::IMREAD_COLOR);
    cv::Size real_size = frame_raw.size();
    const int width = 256, height = 256, size_class = sp_size;
    cv_misc::Chipping chips(real_size, cv::Size(width, height), chip_overlap);

    spt::GSLIC _superpixel({
                                   .img_size = { width, height },
                                   .no_segs = 64,
                                   .spixel_size = size_class,
                                   .no_iters = 5,
                                   .coh_weight = 0.6f,
                                   .do_enforce_connectivity = true,
                                   .color_space = gSLICr::CIELAB,
                                   .seg_method = gSLICr::GIVEN_SIZE
                           });
}

int main(int argc, char* argv[]) {
    ///////////////////////////
    // Argument Parser
    ///////////////////////////
    ArgumentParser parser("Superpixel Feature Inference Pipeline");
    parser.add_argument("-d", "Dataset location", true);
    parser.add_argument("-o", "Output location", true);
    parser.add_argument("-c", "Chipping Overlap (=0.5)");
    parser.add_argument("-s", "Superpixel Size (=32)");
    try {
        parser.parse(argc, argv);
    } catch (const ArgumentParser::ArgumentNotFound& ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    if (parser.is_help()) return 0;

    ///////////////////////////
    // Dataset
    ///////////////////////////
    const fs::path dataset = fs::path(parser.get<std::string>("d"));

    ///////////////////////////
    // Chipping
    ///////////////////////////
    const float chip_overlap = parser.exists("c") ? parser.get<float>("c") : 0.5;

    ///////////////////////////
    // Superpixel
    ///////////////////////////
    const int sp_size = parser.exists("s") ? parser.get<int>("s") : 32;

    ///////////////////////////
    // Process input images
    ///////////////////////////
    for (size_t i = 0; i < images.size(); ++i) {
        const std::string fname(images[i]);
        std::cout << "Processing " << fname << std::endl;
        process_tif(dataset, fname, chip_overlap, sp_size);
    }
}