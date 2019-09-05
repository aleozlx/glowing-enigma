#define BOOST_TEST_MODULE test_pq
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <pqxx/pqxx>

#include "saver.hpp"

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

int test_stream_count(pqxx::connection &conn) {
    int ret;
    pqxx::work w_ct(conn);
    pqxx::result r = w_ct.exec_prepared("sql_ct");
    r.at(0).at(0).to(ret);
    w_ct.commit();
    return ret;
}

/*
CREATE TABLE test_stream (
    name VARCHAR(16),
    feature REAL[]
);
*/
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
        int ct0 = test_stream_count(conn), ct1;
        pqxx::work w_stream(conn);
        pqxx::stream_to s {
            w_stream, "test_stream",
            std::vector<std::string> {
                "name", "feature"
            }
        };
        for(auto const &feature: features) {
            saver::vec2str(feature, feature_buffer);
            s<<std::make_tuple(name, feature_buffer);
        }
        s.complete();
        w_stream.commit();
        ct1 = test_stream_count(conn);
        BOOST_TEST(ct1-ct0 == features.size());
    }
    catch (const std::exception &e) {
        std::cerr<<e.what()<<std::endl;
        BOOST_TEST(false);
    }
}