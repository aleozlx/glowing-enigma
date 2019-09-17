#include <string>
#include <thread>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pqxx/pqxx>
#include <tensorflow/core/public/session.h>
#include <omp.h>
#include "argparse.hpp"
#include "misc_os.hpp"
#include "superpixel.hpp"
#include "dcnn.hpp"
#include "saver.hpp"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace tf = tensorflow;

void process_tif(const fs::path &dataset, const std::string &fname, spt::dnn::IComputeFrameSuperpixel *dcnn, const float chip_overlap, const std::string &dcnn_name, const int sp_size, bool verbose = false) {
    cv::Mat frame_raw = cv::imread(fname, cv::IMREAD_COLOR);
    cv::Size real_size = frame_raw.size();
    const int width = 256, height = 256, size_class = sp_size;
    spt::dnn::Chipping chips(real_size, cv::Size(width, height), chip_overlap);

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

    try{
        pqxx::connection conn("dbname=xview user=postgres");
        fs::path pthFname(fname);
        conn.prepare("sql_find_frame_id", "select id from frame where image = $1");
        conn.prepare("sql_match_bbox2", // (image, cx, cy)
                     "select image, class_label.label_name, bbox.xview_type_id, bbox.xview_bounds_imcoords \
from frame join bbox on frame.id = bbox.frame_id join class_label on bbox.xview_type_id = class_label.id \
where frame.image = $1 and st_point($2, $3) && bbox.xview_bounds_imcoords \
order by st_area(bbox.xview_bounds_imcoords);");
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
        std::cerr<<"Looking up frame_id for "<<image<<std::endl;
        pqxx::result r = w_frame.exec_prepared("sql_find_frame_id", image);
        w_frame.commit();
        if(r.size() == 0) {
            std::cerr<<"frame_id not found."<<std::endl;
            return;
        }

        int frame_id = r[0][0].as<int>();

        cv::Mat frame, superpixel_labels, superpixel_selected, frame_dcnn;
        cv::Rect roi;
        std::vector<std::vector<cv::Point>> superpixel_sel_contour;
        cv::Moments superpixel_moments;
        std::vector<float> superpixel_feature_buffer;
        superpixel_feature_buffer.resize(dcnn->GetFeatureDim() * dcnn->GetNSP());
        std::string superpixel_feature_strbuffer;

        unsigned long ct_superpixel = 0;
        for(int chip_id = 0; chip_id<chips.nchip; ++chip_id) {
            roi = chips.GetROI(chip_id);
            frame = frame_raw(roi);
            spt::ISuperpixel *superpixel = _superpixel.Compute(frame);
            ct_superpixel += superpixel->GetNumSuperpixels();
        }
        std::cout<<"Superpixels to be scanned: "<<ct_superpixel<<std::endl;

        pqxx::connection conn2("dbname=xview user=postgres");
        pqxx::work w_spstream(conn2);
        pqxx::stream_to sps {
                w_spstream, "superpixel_inference",
                std::vector<std::string> {
                        "frame_id",
                        "size_class",
                        "area",
                        "centroid_abs_x",
                        "centroid_abs_y",
                        "dcnn_name",
                        "dcnn_feature",
                        "class_label",
                        "class_label_multiplicity"
                }
        };

        int rows_inserted = 0;
        for(int chip_id = 0; chip_id<chips.nchip; ++chip_id) {
            roi = chips.GetROI(chip_id);

            frame = frame_raw(roi);
            cv::cvtColor(frame, frame_dcnn, cv::COLOR_BGR2RGB);

            spt::ISuperpixel *superpixel = _superpixel.Compute(frame);
            unsigned int nsp = superpixel->GetNumSuperpixels();
            superpixel->GetLabels(superpixel_labels);

            #pragma omp critical(DCNNInference)
            {
                dcnn->Compute(frame_dcnn, superpixel_labels);
                dcnn->GetFeature(superpixel_feature_buffer.data());

            }

            for(unsigned int s = 0; s<nsp; ++s) {
                superpixel_selected = superpixel_labels == s;
                cv::findContours(superpixel_selected, superpixel_sel_contour, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                superpixel_moments = cv::moments(superpixel_sel_contour[0], true);
                const auto area = static_cast<float>(superpixel_moments.m00);
                if (area > 0) {
                    const auto cxf32 = static_cast<float>(superpixel_moments.m10/area+roi.x), cyf32 = static_cast<float>(superpixel_moments.m01/area+roi.y);
                    pqxx::work w_bbox(conn);
                    r = w_bbox.exec_prepared("sql_match_bbox2", image, (int)cxf32, (int)cyf32);
                    w_bbox.commit();
                    int class_label_multiplicity = r.size();
                    if(r.size() > 0) {
                        spt::pgsaver::vec2str(
                                superpixel_feature_buffer,
                                (s*dcnn->GetNSP()),
                                dcnn->GetNSP(),
                                superpixel_feature_strbuffer);

                        sps<<std::make_tuple(
                                frame_id, size_class,
                                area, (int)cxf32, (int)cyf32,
                                dcnn_name, superpixel_feature_strbuffer,
                                r[0]["xview_type_id"].as<int>(), class_label_multiplicity);
                        ++rows_inserted;
                        if (verbose) {
                            std::cout << "<frame_id = " << frame_id << ", chip = " << roi << ", s = " << s << ">"
                                      << std::endl;
                            std::cout << "  Area = " << area << std::endl;
                            std::cout << "  Centroid = " << cxf32 << "," << cyf32 << std::endl;
                            std::cout << "  Objects = " << class_label_multiplicity << std::endl;
                            std::cout << "    ";
                            for (auto const &row: r) {
                                std::cout << row["label_name"] << ". ";
                            }
                            std::cout << std::endl;
                        }
                    }
                }

            }
        }
        sps.complete();
        w_spstream.commit();
        std::cerr<<"Done. +"<<rows_inserted<<" rows"<<std::endl;
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
    parser.add_argument("-n", "DCNN name", true);
    parser.add_argument("-d", "Dataset location", true);
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
    // std::string fname = "/tank/datasets/research/xView/train_images/1036.tif";

    ///////////////////////////
    // Chipping
    ///////////////////////////
    const float chip_overlap = parser.exists("c") ? parser.get<float>("c") : 0.5;

    ///////////////////////////
    // Superpixel
    ///////////////////////////
    const int sp_size = parser.exists("s") ? parser.get<int>("s") : 32;

    ///////////////////////////
    // DCNN Inference (shared across omp threads)
    ///////////////////////////
    const std::string dcnn_name = parser.get<std::string>("n");
    spt::dnn::VGG16SP dcnn;
    dcnn.Summary();
    tf::SessionOptions options;
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.45);
    if(dcnn.NewSession(options)) {
        std::cerr<<"Successfully initialized a new TensorFlow session."<<std::endl;
    }
    else {
        std::cerr<<"Failed to initialized a new TensorFlow session."<<std::endl;
    }
    dcnn.SetInputResolution(256, 256);

    ///////////////////////////
    // Parallel image directory scanning
    ///////////////////////////
    // Limit number of threads because each thread is holding expensive resources
    size_t nproc_omp = std::min(omp_get_max_threads(), 20);
    os_misc::Glob train_images((dataset / "train_images/*.tif").string().c_str());
    #pragma omp parallel for num_threads(nproc_omp) default(none) shared(dataset, train_images, dcnn_name, dcnn)
    for (size_t i = 0; i < train_images.size(); ++i) {
        int tid = omp_get_thread_num();
        std::string fname(train_images[i]);
        std::cout << "tid=" << tid << " Processing " << fname << std::endl;
        process_tif(dataset, fname, &dcnn, chip_overlap, dcnn_name, sp_size);
    }
    return 0;
}
