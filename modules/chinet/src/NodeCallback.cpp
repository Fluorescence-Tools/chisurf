#include <NodeCallback.h>
#include "info.h"
#include <climits>

//using namespace rttr;

template <typename T>
inline void mul(
        T* tmp,
        size_t &n_elements,
        const std::map<std::string, std::shared_ptr<Port>> &inputs
        )
{
    std::fill(tmp, tmp + n_elements, 1.0);
    for(auto &o : inputs){
        T* va; int vn;
        o.second->get_value<T>(&va, &vn);
        for(int i=0; i < n_elements; i++){
            tmp[i] *= va[i];
        }
    }
}

template <typename T>
inline void add(
        T* tmp,
        size_t &n_elements,
        const std::map<std::string, std::shared_ptr<Port>> &inputs
)
{
    std::fill(tmp, tmp + n_elements, 1.0);
    for(auto &o : inputs){
        T* va; int vn;
        o.second->get_value<T>(&va, &vn);
        for(size_t i=0; i < n_elements; i++){
            tmp[i] += va[i];
        }
    }
}


template <typename T>
void combine(
        std::map<std::string, std::shared_ptr<Port>> &inputs,
        std::map<std::string, std::shared_ptr<Port>> &outputs,
        int operation
        ){
    if (is_chinet_verbose()) {
        std::clog << "-- Combining values of input ports."  << std::endl;
    }
    size_t n_elements = UINT_MAX;
    if (is_chinet_verbose()) {
        std::clog << "-- Determining input vector with smallest length." << std::endl;
    }
    for(auto &o : inputs){
        n_elements = std::min(n_elements, o.second->current_size());
    }
    if (is_chinet_verbose()) {
        std::clog << "-- The smallest input vector has a length of: " << n_elements << std::endl;
    }
    auto tmp = (T*) malloc(n_elements * sizeof(T));
    if (tmp == nullptr) {
        throw std::runtime_error("Memory allocation failed in combine function");
    }
    switch(operation){
        case 0:
            if (is_chinet_verbose()) {
                std::clog << "-- Adding input ports" << std::endl;
            }
            add(tmp, n_elements, inputs);
            break;
        case 1:
            if (is_chinet_verbose()) {
                std::clog << "-- Multiplying input ports" << std::endl;
            }
            mul(tmp, n_elements, inputs);
            break;
        default:
            break;
    }
    if(!outputs.empty()){
        if (outputs.find("outA") == outputs.end() ) {
            if (is_chinet_verbose()) {
                std::clog << "ERROR: Node does not define output port with the name 'outA' " << std::endl;
            }
            outputs.begin()->second->set_value(tmp, n_elements);
        } else {
            if (is_chinet_verbose()) {
                std::clog << "Setting value to output " << std::endl;
            }
            outputs["outA"]->set_value(tmp, n_elements);
        }
    } else{
        std::cerr << "ERROR: There are no output ports." << std::endl;
    }
    free(tmp);
}


template <typename T>
void addition(
        std::map<std::string, std::shared_ptr<Port>> &inputs,
        std::map<std::string, std::shared_ptr<Port>> &outputs
)
{
    if (is_chinet_verbose()) {
        std::clog << "addition"  << std::endl;
    }
    combine<T>(inputs, outputs, 0);
}

template <typename T>
void multiply(
        std::map<std::string, std::shared_ptr<Port>> &inputs,
        std::map<std::string, std::shared_ptr<Port>> &outputs
)
{
    combine<T>(inputs, outputs, 1);
}

void nothing(
        std::map<std::string, std::shared_ptr<Port>> &inputs,
        std::map<std::string, std::shared_ptr<Port>> &outputs){
    if (is_chinet_verbose()) {
        std::clog << "[NodeCallback::nothing] called; inputs=" << inputs.size() << ", outputs=" << outputs.size() << std::endl;
    }
}

void passthrough(
        std::map<std::string, std::shared_ptr<Port>> &inputs,
        std::map<std::string, std::shared_ptr<Port>> &outputs
        ){
    if (is_chinet_verbose()) {
        std::clog << "[NodeCallback::passthrough] inputs=" << inputs.size() << ", outputs=" << outputs.size() << std::endl;
    }
    for(auto it_in = inputs.cbegin(), end_in = inputs.cend(),
            it_out = outputs.cbegin(), end_out = outputs.cend();
            it_in != end_in || it_out != end_out;)
    {
        double* va; int vn;
        it_in->second->get_value(&va, &vn);
        auto v = std::vector<double>();
        v.assign(va, va + vn);
        if(it_in != end_in) {
            ++it_in;
        } else{
            break;
        }
        if(it_out != end_out) {
            it_out->second->set_value(v.data(), v.size());
            ++it_out;
        } else{
            break;
        }
    }
}

void convolve_sum_of_exponentials_periodic(
        std::map<std::string, Port*> &inputs,
        std::map<std::string, Port*> &outputs
        ){
    if (is_chinet_verbose()) {
        std::clog << "[NodeCallback::convolve_sum_of_exponentials_periodic] called; inputs=" << inputs.size() << ", outputs=" << outputs.size() << std::endl;
    }
    // Make sure that all inputs are set

    int n_irf; double* irf;
    inputs["irf"]->get_value(&irf, &n_irf);
    int n_decay; double* decay;
    inputs["decay"]->get_value(&decay, &n_decay);
    int n_lifetimes; double* lifetimes;
    inputs["lifetimes"]->get_value(&lifetimes, &n_lifetimes);

//    long start = inputs["start"]->get_value<long>();
//    long stop = inputs["stop"]->get_value<long>();
//    double dt = inputs["dt"]->get_value<double>();
//    long period = inputs["period"]->get_value<long>();
//
//
//    Functions::convolve_sum_of_exponentials_periodic(
//            decay.data(), decay.size(),
//            lifetimes.data(), lifetimes.size(),
//            irf.data(), irf.size(),
//            start, stop, dt, period
//            );
//
//    output->set_slot_value("decay", decay);
}

void AV(
        std::map<std::string, Port*> &inputs,
        std::map<std::string, Port*> &output){
    /*
    auto g = dyeDensity(

            );
    */
}
