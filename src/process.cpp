#include <string>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pqxx/pqxx>
#include "argparse.hpp"
#include "superpixel.hpp"
#include "dcnn.hpp"
#include "saver.hpp"

extern "C" {
#include <glob.h>
}

namespace fs = std::filesystem;

void init() {
    // run as: [program name] "0 -c" abc -a 1 -sdfl --flag -v 1 2.7 3 4 9 8.12 87
    // [program name] -sdfv 1 -o "C:\Users\User Name\Directory - Name\file.dat" "C:\Users\User Name 2\Directory 2 - Name 2\file2.dat" C:/tmp/tmp.txt

    // ArgumentParser parser("Argument parser example");
    // parser.add_argument("-a", "an integer");
    // parser.add_argument("-s", "an combined flag", true);
    // parser.add_argument("-d", "an combined flag", true);
    // parser.add_argument("-f", "an combined flag", true);
    // parser.add_argument("--flag", "a flag");
    // parser.add_argument("-v", "a vector", true);
    // parser.add_argument("-l", "--long", "a long argument", false);
    // parser.add_argument("--files", "input files", false);
    // try {
    //     parser.parse(argc, argv);
    // } catch (const ArgumentParser::ArgumentNotFound& ex) {
    //     std::cout << ex.what() << std::endl;
    //     return 0;
    // }
    // if (parser.is_help()) return 0;
    // std::cout << "a: " << parser.get<int>("a") << std::endl;
    // std::cout << "flag: " << std::boolalpha << parser.get<bool>("flag")
    //         << std::endl;
    // std::cout << "d: " << std::boolalpha << parser.get<bool>("d") << std::endl;
    // std::cout << "long flag: " << std::boolalpha << parser.get<bool>("l")
    //         << std::endl;
    // auto v = parser.getv<double>("v");
    // std::cout << "v: ";
    // std::copy(v.begin(), v.end(), std::ostream_iterator<double>(std::cout, " "));
    // double sum;
    // for (auto& d : v) sum += d;
    // std::cout << " sum: " << sum << std::endl;
    // auto f = parser.getv<std::string>("files");
    // std::cout << "files: ";
    // std::copy(f.begin(), f.end(),
    //         std::ostream_iterator<std::string>(std::cout, " | "));
    // std::cout << std::endl;
    // f = parser.getv<std::string>("");
    // std::cout << "free args: ";
    // std::copy(f.begin(), f.end(),
    //         std::ostream_iterator<std::string>(std::cout, " "));
    // std::cout << std::endl;
}

void process_tif(const std::string &dataset, const std::string &fname) {
    cv::Mat frame_raw = cv::imread(fname, cv::IMREAD_COLOR);
    cv::Size real_size = frame_raw.size();
    const int width = 256, height = 256, size_class = 32;
    spt::dnn::Chipping chips(real_size, cv::Size(width, height), 0.5); // TODO overlap as argument

    spt::dnn::VGG16SP dcnn;
//    dcnn.Summary();
    if(dcnn.NewSession()) {
        std::cerr<<"Successfully initialized a new TensorFlow session."<<std::endl;
    }
    else {
        std::cerr<<"Failed to initialized a new TensorFlow session."<<std::endl;
    }
    dcnn.SetInputResolution(256, 256);

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
        std::string image = fs::path(fname).lexically_relative(dataset).string();
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
        superpixel_feature_buffer.resize(dcnn.GetFeatureDim());
        std::string superpixel_feature_strbuffer;

        unsigned long ct_superpixel = 0;
        for(int chip_id = 0; chip_id<chips.nchip; ++chip_id) {
            roi = chips.GetROI(chip_id);
            frame = frame_raw(roi);
            spt::ISuperpixel *superpixel = _superpixel.Compute(frame);
            ct_superpixel += superpixel->GetNumSuperpixels();
        }
        std::cout<<"Estimated number of superpixels: "<<ct_superpixel<<std::endl;

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
        std::string dcnn_name = "VGG16SP_test";
        int rows_inserted = 0;
        for(int chip_id = 0; chip_id<chips.nchip; ++chip_id) {
            roi = chips.GetROI(chip_id);

            frame = frame_raw(roi);
            cv::cvtColor(frame, frame_dcnn, cv::COLOR_BGR2RGB);

            spt::ISuperpixel *superpixel = _superpixel.Compute(frame);
            unsigned int nsp = superpixel->GetNumSuperpixels();
            superpixel->GetLabels(superpixel_labels);

            dcnn.Compute(frame_dcnn, superpixel_labels);
            for(int s = 0; s<nsp; ++s) {
                superpixel_selected = superpixel_labels == s;
                cv::findContours(superpixel_selected, superpixel_sel_contour, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                superpixel_moments = cv::moments(superpixel_sel_contour[0], true);
#define v0 superpixel_moments.m00
#define v1 superpixel_moments.mu02
#define v2 superpixel_moments.mu20
#define v3 superpixel_moments.mu11
                if (v0 > 0) {
                    dcnn.GetFeature(s, superpixel_feature_buffer.data());
                    float cxf32 = superpixel_moments.m10/v0+roi.x, cyf32 = superpixel_moments.m01/v0+roi.y;
                    pqxx::work w_bbox(conn);
                    r = w_bbox.exec_prepared("sql_match_bbox2", image, (int)cxf32, (int)cyf32);
                    w_bbox.commit();
                    int class_label_multiplicity = r.size();
                    if(r.size() > 0) {
                        spt::pgsaver::vec2str(superpixel_feature_buffer, superpixel_feature_strbuffer);

                        sps<<std::make_tuple(
                                frame_id, size_class,
                                v0, (int)cxf32, (int)cyf32,
                                dcnn_name, superpixel_feature_strbuffer,
                                r[0]["xview_type_id"].as<int>(), class_label_multiplicity);
                        ++rows_inserted;
//                        std::cout<<"<frame_id = "<<frame_id<<", chip = "<<roi<<", s = "<<s<<">"<<std::endl;
//                        std::cout<<"  Area = "<<v0<<std::endl;
//                        std::cout<<"  Centroid = "<<cxf32<<","<<cyf32<<std::endl;
//                        std::cout<<"  Objects = "<<class_label_multiplicity<<std::endl;
//                        std::cout<<"    ";
//                        for(auto row: r) {
//                            std::cout<<row["label_name"]<<". ";
//                        }
//                        std::cout<<std::endl;
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

class Glob {
protected:
    glob_t glob_result;
    bool _error = false;
public:
    Glob(const char *pattern) {
        std::memset(&glob_result, 0, sizeof(glob_result));
        if(glob(pattern, GLOB_TILDE, NULL, &glob_result)!=0) {
            std::cerr << "glob() error" << std::endl;
            _error = true;
        }
    }

    ~Glob() {
        globfree(&glob_result);
    }

    size_t size() {
        return _error ? 0 : glob_result.gl_pathc;
    }

    const char* operator[](int i) {
        return glob_result.gl_pathv[i];
    }
};

int main(int argc, char* argv[]) {
    std::string dataset = "/tank/datasets/research/xView";
//    std::string fname = "/tank/datasets/research/xView/train_images/1036.tif";

    Glob train_images("/tank/datasets/research/xView/train_images/3*.tif");
    for(size_t i = 0; i < train_images.size(); ++i) {
        std::string fname(train_images[i]);
        std::cout<<"Processing "<<fname<<std::endl;
        process_tif(dataset, fname);
    }

    return 0;
}
