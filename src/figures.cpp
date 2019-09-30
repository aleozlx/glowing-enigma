#include <string>
#include <thread>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pqxx/pqxx>
#include <omp.h>
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

void process_tif(const fs::path &dataset, const std::string &fname, const fs::path &output, const float chip_overlap, const int sp_size, bool verbose = false) {
    cv::Mat frame_raw = cv::imread(fname, cv::IMREAD_COLOR);
    cv::Size real_size = frame_raw.size();
    const int width = 385, height = 385, size_class = sp_size;
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

    try {
        pqxx::connection conn("dbname=xview user=postgres");
        fs::path pthFname(fname);
        conn.prepare("sql_find_frame_id", "select id from frame where image = $1");
#if 0
        conn.prepare("sql_match_bbox2_ct", // (image, cx, cy)
                     "select count(*) \
from frame join bbox on frame.id = bbox.frame_id join class_label cls on bbox.xview_type_id = cls.id \
where frame.image = $1 and st_point($2, $3) && bbox.xview_bounds_imcoords;");
        conn.prepare("sql_bbox_in_view", // (image, cx, cy)
                     "select image, cls.label_name, bbox.xview_type_id, ST_XMin(bbox.xview_bounds_imcoords) as xmin, ST_YMin(bbox.xview_bounds_imcoords) as ymin, ST_XMax(bbox.xview_bounds_imcoords) as xmax, ST_YMax(bbox.xview_bounds_imcoords) as ymax \
from frame join bbox on frame.id = bbox.frame_id join class_label cls on bbox.xview_type_id = cls.id \
where frame.image = $1 and ST_MakeEnvelope($2, $3, $4, $5) && bbox.xview_bounds_imcoords;");
#else
        // Sub-selection
        conn.prepare("sql_match_bbox2_ct", // (image, cx, cy)
                     "select count(*) \
from frame join bbox on frame.id = bbox.frame_id join class_label cls on bbox.xview_type_id = cls.id \
where frame.image = $1 AND cls.label_name in ('Tower','Fixed-wing Aircraft', 'Yacht', 'Passenger Vehicle', 'Shipping Container') and st_point($2, $3) && bbox.xview_bounds_imcoords;");
        conn.prepare("sql_bbox_in_view", // (image, cx, cy)
                     "select image, cls.label_name, bbox.xview_type_id, ST_XMin(bbox.xview_bounds_imcoords) as xmin, ST_YMin(bbox.xview_bounds_imcoords) as ymin, ST_XMax(bbox.xview_bounds_imcoords) as xmax, ST_YMax(bbox.xview_bounds_imcoords) as ymax \
from frame join bbox on frame.id = bbox.frame_id join class_label cls on bbox.xview_type_id = cls.id \
where frame.image = $1 AND cls.label_name in ('Tower','Fixed-wing Aircraft', 'Yacht', 'Passenger Vehicle', 'Shipping Container') and ST_MakeEnvelope($2, $3, $4, $5) && bbox.xview_bounds_imcoords;");
#endif
        pqxx::work w_frame(conn);
#if __has_include(<filesystem>)
        std::string image = fs::path(fname).lexically_relative(dataset).string();
#else
        // cuz path lib in C++ is obnoxious af.
        std::string image = fname;
        {
            std::string _dataset = dataset.string() + "/";
            if(fname.find(_dataset) == 0)
                image.erase(0, _dataset.length());
        }
#endif
        std::cerr << "Looking up frame_id for " << image << std::endl;
        pqxx::result r = w_frame.exec_prepared("sql_find_frame_id", image);
        w_frame.commit();
        if (r.empty()) {
            std::cerr << "frame_id not found." << std::endl;
            return;
        }

        int frame_id = r[0][0].as<int>();

        cv::Mat frame, frame_rgb, frame_rgb2, superpixel_labels, superpixel_selected, superpixel_contour, im_save;
        std::vector<std::vector<cv::Point>> superpixel_sel_contour;
        cv::Rect roi;
        cv::Moments superpixel_moments;

        unsigned long ct_superpixel = 0;
        for(int chip_id = 0; chip_id<chips.nchip; ++chip_id) {
            roi = chips.GetROI(chip_id);
            frame = frame_raw(roi);
            spt::ISuperpixel *superpixel = _superpixel.Compute(frame);
            ct_superpixel += superpixel->GetNumSuperpixels();
        }
        std::cout<<"Superpixels to be scanned: "<<ct_superpixel<<std::endl;

        const cv::Scalar color_superpixel(200, 5, 240), color_bbox(240, 240, 5);

        for(int chip_id = 0; chip_id<chips.nchip; ++chip_id) {
            roi = chips.GetROI(chip_id);

            frame = frame_raw(roi);
            cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
            frame_rgb2 = frame_rgb.clone();
            spt::ISuperpixel *superpixel = _superpixel.Compute(frame_rgb);
            unsigned int nsp = superpixel->GetNumSuperpixels();
            superpixel->GetLabels(superpixel_labels);
            superpixel->GetContour(superpixel_contour);

            char cstr_fname_out[200];

            // Save input image
            if (sp_size == 8) { // TODO change to tid
                std::snprintf(cstr_fname_out, 200, "f%dc%di.png", frame_id, chip_id);
                cv::imwrite((output / cstr_fname_out).string(), frame);
            }

            // Draw superpixels
            frame_rgb.setTo(color_superpixel, superpixel_contour);

            // Draw labelled bounding boxes
            pqxx::work w_bbox(conn);
            pqxx::result r = w_bbox.exec_prepared("sql_bbox_in_view", image, roi.x, roi.y, roi.x+roi.width, roi.y+roi.height);
            w_bbox.commit();
            for (auto const &row: r) {
                const int xmin = row["xmin"].as<int>()-roi.x,
                    ymin = row["ymin"].as<int>()-roi.y,
                    xmax = row["xmax"].as<int>()-roi.x,
                    ymax = row["ymax"].as<int>()-roi.y;
                const cv::Point a(xmin, ymin), b(xmax, ymax);
                cv::rectangle(frame_rgb, a, b, color_bbox, 2);
            }

            cv::cvtColor(frame_rgb, im_save, cv::COLOR_RGB2BGR);

            std::snprintf(cstr_fname_out, 200, "f%dc%ds%d.png", frame_id, chip_id, (int)sp_size);
            std::string fname_out = output / std::string(cstr_fname_out);
            cv::imwrite(fname_out, im_save);

            // Draw superpixel labels
            int total_match = 0;
            for(unsigned int s = 0; s<nsp; ++s) {
                superpixel_selected = superpixel_labels == s;
                cv::findContours(superpixel_selected, superpixel_sel_contour, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
                superpixel_moments = cv::moments(superpixel_sel_contour[0], true);
                const auto area = static_cast<float>(superpixel_moments.m00);
                if (area > 0) {
                    const auto cxf32 = static_cast<float>(superpixel_moments.m10/area+roi.x), cyf32 = static_cast<float>(superpixel_moments.m01/area+roi.y);
                    pqxx::work w_bbox_match(conn);
                    r = w_bbox_match.exec_prepared("sql_match_bbox2_ct", image, (int)cxf32, (int)cyf32);
                    w_bbox.commit();
                    const auto ct_match = r[0][0].as<int>();
                    if (ct_match > 0) {
                        cv::drawContours(frame_rgb2, superpixel_sel_contour, 0, color_bbox, 1);
                        ++total_match;
                    }
                }
            }

            if (total_match > 0) {
                cv::cvtColor(frame_rgb2, im_save, cv::COLOR_RGB2BGR);
                std::snprintf(cstr_fname_out, 200, "f%dc%ds%dm.png", frame_id, chip_id, (int) sp_size);
                cv::imwrite((output / cstr_fname_out).string(), im_save);
            }
        }
    }
    catch (const std::exception &e) {
        std::cerr<<e.what()<<std::endl;
    }

}

int main(int argc, char* argv[]) {
    ///////////////////////////
    // Argument Parser
    ///////////////////////////
    ArgumentParser parser("Superpixel Feature Inference Pipeline");
    parser.add_argument("-d", "Dataset location", true);
    parser.add_argument("-o", "Output location", true);
    parser.add_argument("-c", "Chipping Overlap (=0.5)");
    try {
        parser.parse(argc, argv);
    } catch (const ArgumentParser::ArgumentNotFound& ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    if (parser.is_help()) return 0;

    ///////////////////////////
    // Dataset & Output
    ///////////////////////////
    const fs::path dataset = fs::path(parser.get<std::string>("d"));
    const fs::path output = fs::path(parser.get<std::string>("o"));

    ///////////////////////////
    // Chipping
    ///////////////////////////
    const float chip_overlap = parser.exists("c") ? parser.get<float>("c") : 0.5;

//    ///////////////////////////
//    // Superpixel
//    ///////////////////////////
//    const int sp_size = parser.exists("s") ? parser.get<int>("s") : 32;

    ///////////////////////////
    // Process input images
    ///////////////////////////
    const std::vector<int> sp_sizes = {8, 18, 24};
//    const std::vector<int> sp_sizes = {6, 8, 12, 18, 24, 32};

    for (size_t i = 0; i < images.size(); ++i) {
        const std::string fname = (dataset/images[i]).string();
        std::cout << "Processing " << fname << std::endl;
        #pragma omp parallel for default(none) shared(dataset, fname, output, sp_sizes)
        for (size_t j = 0; j < sp_sizes.size(); ++j)
            process_tif(dataset, fname, output, chip_overlap, sp_sizes[j]);
    }
}