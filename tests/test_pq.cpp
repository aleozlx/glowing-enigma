#define BOOST_TEST_MODULE test_pq
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <pqxx/pqxx>

extern "C" {
#include "fpconv.h"
}

std::string connection = "dbname=xview user=postgres";

BOOST_AUTO_TEST_CASE(test_pq_conn) {
    bool status = false;
    try {
        pqxx::connection conn(connection);
        status = conn.is_open();
    }
    catch (const std::exception &e) {
        std::cerr<<e.what()<<std::endl;
    }
    BOOST_TEST(status);
}

#define dtoa_(fp, dest) fpconv_dtoa(fp, dest)

template<typename F>
size_t vec2str(size_t dim, F *vec, char *dst) {
    if(dim<=0) return 0;
    char *dst0 = dst;
    const char header[] = "array[";
    std::strcpy(dst, header);
    dst += std::strlen(header);
    size_t len;
    do {
        len = dtoa_(static_cast<double>(*vec), dst);
        dst[len] = ',';
        dst += len+1;
        vec++;
    } while(--dim);
    *(dst-1) = ']';
    *dst = '\0';
    return dst-dst0;
}

BOOST_AUTO_TEST_CASE(test_stream_to) {
    std::string name = "a";
    std::vector<std::vector<double>> features {
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
    };
    std::vector<double> ff {1.0, 2.0, 1.0};
    std::tuple<std::string, std::vector<double>> r = {"a", ff};
    std::string feature_buffer;
    try {
        pqxx::connection conn(connection);
        pqxx::work cur(conn);
        pqxx::stream_to s {
            cur, "test_stream",
            std::vector<std::string> {
                "name", "feature"
            }
        };
        for(auto const &feature: features) {
            feature_buffer.resize(60*4096+20);
            size_t len = vec2str(3, feature.data(), const_cast<char*>(feature_buffer.data()));
            feature_buffer.resize(len);
//            std::cout<<feature_buffer<<std::endl;
        }
        s.complete();
        cur.commit();
    }
    catch (const std::exception &e) {
        std::cerr<<e.what()<<std::endl;
    }
}