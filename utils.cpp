#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "utils.h"
#include "nlohmann/json.hpp"


nlohmann::json read_json_file(std::string filename) {
    std::ifstream f(filename);
    nlohmann::json data = nlohmann::json::parse(f);
    f.close();
    return data;
}

Instance import_instance(std::string filename, std::ostream& out_stream = std::cout) {
    nlohmann::json inst_descriptor = read_json_file(filename);
    
    Instance inst_object;

    inst_object.nb_jobs       = inst_descriptor["parameters"]["size"]["nb_jobs"];
    inst_object.nb_tasks      = inst_descriptor["parameters"]["size"]["nb_tasks"];
    inst_object.nb_machines   = inst_descriptor["parameters"]["size"]["nb_machines"];
    inst_object.nb_operators  = inst_descriptor["parameters"]["size"]["nb_operators"];

    inst_object.unit_penalty  = inst_descriptor["parameters"]["costs"]["unit_penalty"];
    inst_object.tardiness     = inst_descriptor["parameters"]["costs"]["tardiness"];
    inst_object.interim       = inst_descriptor["parameters"]["costs"]["interim"];


    // Iterate over tasks, import and print details
    for (auto& task_descriptor : inst_descriptor["tasks"]) {
        Task task_object;
        task_object.id = task_descriptor["task"];
        task_object.processing_time = task_descriptor["processing_time"];

        task_object.machines = std::vector<int>();
        for (auto& machine : task_descriptor["machines"]) {
            std::vector<int> qualified_operators = std::vector<int>();
            for (auto& op : machine["operators"]) {
                qualified_operators.push_back(op);
            }
            task_object.operators.push_back(qualified_operators);
            task_object.machines.push_back(machine["machine"]);
        }
        
        // Commit log task details
        out_stream << "== Imported task " << task_object.id << " ==" << std::endl;
        out_stream << "* processing time: " << task_object.processing_time << std::endl;
        out_stream << "* possible supports: " << std::endl;
        for (int m = 0; m < int(task_object.machines.size()); m++) {
            out_stream << "  M" << task_object.machines[m] << " with O{";
            for (size_t o = 0; o < task_object.operators[m].size(); o++) {
            out_stream << task_object.operators[m][o];
            if (o < task_object.operators[m].size() - 1) {
                out_stream << ",";
            }
            }
            out_stream << "}" << std::endl;
        }
        out_stream << std::endl;

        // Commit task to instance
        inst_object.tasks.push_back(task_object);
    }

    out_stream << std::endl;
    // Iterate over jobs, import and print details
    int job_idx = 0;
    for (auto& job_descriptor : inst_descriptor["jobs"]) {
        Job job_object;
        job_object.id = job_descriptor["job"];
        job_object.release_date = job_descriptor["release_date"];
        job_object.due_date     = job_descriptor["due_date"];
        job_object.weight       = job_descriptor["weight"];
        job_object.sequence     = std::vector<int>();

        for (int task_id : job_descriptor["sequence"]) {
            job_object.sequence.push_back(task_id - 1); // ids are offset by 1 in the JSON file
            inst_object.tasks[task_id - 1].job_parent = job_idx;
        }
        
        // Commit log job details
        out_stream << "== Imported job " << job_object.id << " ==" << std::endl;
        out_stream << "* release date: " << job_object.release_date << std::endl;
        out_stream << "* due date: " << job_object.due_date << std::endl;
        out_stream << "* weight: " << job_object.weight << std::endl;
        out_stream << "* task sequence: ";
        for (auto& task_idx : job_object.sequence) {
            out_stream << task_idx + 1 << "->"; // ids are offset by 1 in the JSON file
        }
        out_stream << "end" << std::endl << std::endl;

        inst_object.jobs.push_back(job_object);
        job_idx++;
    }



    return inst_object;
}