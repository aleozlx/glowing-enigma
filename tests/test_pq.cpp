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
#define MAX_LEN_DTOA (24)
#define MAX_DIM_FEATURE (4096)

template<typename F>
size_t vec2str(size_t dim, F *vec, char *dst) {
    if(dim<=0) return 0;
    char *dst0 = dst;
    *dst++ = '{';
    do {
        dst += dtoa_(static_cast<double>(*vec), dst);
        *dst++ = ',';
        vec++;
    } while(--dim);
    *(dst-1) = '}';
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
    std::string feature_buffer;
    try {
        pqxx::connection conn(connection);
        conn.prepare("sql_ct", "select count(*) from test_stream");
        int ct0, ct1;
        {
            pqxx::work w_ct(conn);
            pqxx::result r = w_ct.exec_prepared("sql_ct");
            r.at(0).at(0).to(ct0);
            w_ct.commit();
        }
        pqxx::work w_stream(conn);
        pqxx::stream_to s {
            w_stream, "test_stream",
            std::vector<std::string> {
                "name", "feature"
            }
        };
        for(auto const &feature: features) {
            feature_buffer.resize(MAX_LEN_DTOA*MAX_DIM_FEATURE+20);
            size_t len = vec2str(3, feature.data(), const_cast<char*>(feature_buffer.data()));
            feature_buffer.resize(len);
            s<<std::make_tuple(name, feature_buffer);
        }
        s.complete();
        w_stream.commit();
        {
            pqxx::work w_ct(conn);
            pqxx::result r = w_ct.exec_prepared("sql_ct");
            r.at(0).at(0).to(ct1);
            w_ct.commit();
        }
        BOOST_TEST(ct1-ct0 == features.size());
    }
    catch (const std::exception &e) {
        std::cerr<<e.what()<<std::endl;
        BOOST_TEST(false);
    }
}