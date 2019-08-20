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

struct Chipping {
    int width, height;
    int chip_width, chip_height;
    int nx, ny, nchip;

    Chipping() { }

    Chipping(cv::Size input_size, cv::Size chip_size) {
        width = input_size.width;
        height = input_size.height;
        chip_width = chip_size.width;
        chip_height = chip_size.height;
        nx = width / chip_width;
        ny = height / chip_height;
        nchip = nx * ny;
    }

    cv::Rect GetROI(int chip_id) {
        // TODO add optional overlap
        int offset_x = chip_id % nx * chip_height;
        int offset_y = chip_id / nx * chip_width;
        return cv::Rect(offset_x, offset_y, chip_width, chip_height);
    }
};

int main(int argc, char* argv[]) {
    std::string dataset = "/tank/datasets/research/xView";
    std::string fname = "/tank/datasets/research/xView/train_images/1036.tif";
    cv::Mat frame_raw = cv::imread(fname, cv::IMREAD_COLOR);
    cv::Size real_size = frame_raw.size();
    const int width = 256, height = 256, size_class = 32;
    Chipping chips(real_size, cv::Size(width, height));
    cv::Mat frame = frame_raw(chips.GetROI(0));
    GSLIC _superpixel({
        .img_size = { width, height },
        .no_segs = 64,
        .spixel_size = size_class,
        .no_iters = 5,
        .coh_weight = 0.6f,
        .do_enforce_connectivity = true,
        .color_space = gSLICr::CIELAB,
        .seg_method = gSLICr::GIVEN_SIZE
    });
    VGG16SP dcnn;
    dcnn.Summary();
    dcnn.NewSession();
    ISuperpixel *superpixel = _superpixel.Compute(frame);
    unsigned int nsp = superpixel->GetNumSuperpixels();
    try{
        pqxx::connection conn("dbname=xview user=postgres");
        fs::path pthFname(fname);
        conn.prepare("sql_find_frame_id", "select id from frame where image = $1");
        conn.prepare("sql_match_bbox", // (image, cx, cy)
"select image, class_label.label_name, bbox.xview_bounds_imcoords \
from frame join bbox on frame.id = bbox.frame_id join class_label on bbox.xview_type_id = class_label.id \
where frame.image = $1 \
    and bbox.xview_bounds_imcoords[1] < $2 and bbox.xview_bounds_imcoords[2] < $3 and bbox.xview_bounds_imcoords[3] > $2 and bbox.xview_bounds_imcoords[4] > $3 \
order by (bbox.xview_bounds_imcoords[3]-bbox.xview_bounds_imcoords[1])*(bbox.xview_bounds_imcoords[4]-bbox.xview_bounds_imcoords[2]);");
        pqxx::work cur(conn);
        std::string image = fs::path(fname).lexically_relative(dataset).string();
        pqxx::result r = cur.prepared("sql_find_frame_id")(image).exec();
        cur.commit();
        if(r.size() == 0)
            return 1;
        int frame_id = r[0][0].as<int>();

        cv::Mat superpixel_labels, superpixel_selected;
        std::vector<std::vector<cv::Point>> superpixel_sel_contour;
        cv::Moments superpixel_moments;
        superpixel->GetLabels(superpixel_labels);
        for(int s = 0; s<nsp; ++s) {
            // create table superpixel_inference (
            //     id SERIAL PRIMARY KEY,
            //     ------------------
            //     -- Superpixel Generation
            //     ------------------
            //     frame_id INT REFERENCES frame(id) NOT NULL,
            //     size_class INT NOT NULL,

            //     ------------------
            //     -- Moments
            //     ------------------
            //     area FLOAT,
            //     centroid_abs_x INT,
            //     centroid_abs_y INT,

            //     ------------------
            //     -- DCNN Feature
            //     ------------------
            //     dcnn_name VARCHAR[16],
            //     dcnn_feature FLOAT[],

            //     ------------------
            //     -- Training Data
            //     ------------------
            //     class_label INT, -- the label of the smallest bounding box containing the centroid
            //     class_label_multiplicity INT -- the number of bounding boxes that the centroid hits
            // );

            superpixel_selected = superpixel_labels == s;
            cv::findContours(superpixel_selected, superpixel_sel_contour, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            superpixel_moments = cv::moments(superpixel_sel_contour[0], true);
            #define v0 superpixel_moments.m00
            #define v1 superpixel_moments.mu02
            #define v2 superpixel_moments.mu20
            #define v3 superpixel_moments.mu11
            if (v0 > 0) {
                std::cout<<"<frame_id = "<<frame_id<<", s = "<<s<<">"<<std::endl;
                std::cout<<"  Area = "<<v0<<std::endl;
                float cxf32 = superpixel_moments.m10/v0, cyf32 = superpixel_moments.m01/v0;
                std::cout<<"  Centroid = "<<cxf32<<","<<cyf32<<std::endl;
                pqxx::work cur(conn);
                r = cur.prepared("sql_match_bbox")(image)((int)cxf32)((int)cyf32).exec();
                cur.commit();
                int class_label_multiplicity = r.size();
                std::cout<<"  Objects = "<<class_label_multiplicity<<std::endl;
                if(r.size() > 0) {
                    std::cout<<"    ";
                    for(auto row: r) {
                        std::cout<<row["label_name"]<<". ";
                    }
                    std::cout<<std::endl;
                }
            }
            
        }
    }
    catch (const std::exception &e) {
        std::cerr<<e.what()<<std::endl;
    }
    return 0;
}
