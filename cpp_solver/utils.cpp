#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <deque>
#include <fstream>
#include <cstdlib>
#include "utils.h"
#include "json.hpp"


nlohmann::json read_json_file(std::string filename) {
    std::ifstream f(filename);
    nlohmann::json data = nlohmann::json::parse(f);
    f.close();
    return data;
}

void print_set(std::set<int> s, int offset, std::ostream& out_stream) {
    out_stream << "{";
    for (auto& e : s) {
        out_stream << e + offset;
        if (e != *s.rbegin()) {
            out_stream << ",";
        }
    }
    out_stream << "}";
}


void print_sequence(std::vector<int> v, int offset, std::ostream& out_stream) {
        out_stream << "[";
        for (auto& t : v) {
            out_stream << t + offset;
            if (t != v.back()) {
                out_stream << "->";
            }
        }
        out_stream << "]";
}



void display_cstr_matrix(std::map<int, std::map<int, int>>& matrix, std::ostream& out_stream) {
    // Print column headers
    out_stream << "   ";
    for (const auto& col : matrix.begin()->second) {
        out_stream << std::setw(4) << col.first + 1;
    }
    out_stream << std::endl;

    // Print rows
    for (const auto& row : matrix) {
        out_stream << std::setw(4) << row.first + 1 << ": ";
        for (const auto& cell : row.second) {
            if (cell.second == 3) {
                out_stream << "\u00d7   ";
            } else if (cell.second == 2) {
                out_stream << "o   ";
            } 
            else if (cell.second == 1) {
                out_stream << "m   ";
            }
            else {
                out_stream << "    "; // Two spaces for proper alignment
            }
        }
        out_stream << std::endl; // New line after each row
    }
}


void displayMatrix(const std::map<int, std::map<int, int>>& matrix, std::ostream& out_stream) {
    if (matrix.empty()) {
        out_stream << "Matrix is empty.\n";
        return;
    }

    // Determine the set of all column headers (keys of the innermost maps)
    std::set<int> columns;
    for (const auto& [outer_key, inner_map] : matrix) {
        for (const auto& [inner_key, _] : inner_map) {
            columns.insert(inner_key);
        }
    }

    // Print the header row
    out_stream << std::setw(4) << " "; // Leave space for row headers
    for (int col : columns) {
        out_stream << std::setw(4) << col;
    }
    out_stream << std::endl;

    // Print each row
    for (const auto& [row_key, inner_map] : matrix) {
        out_stream << std::setw(4) << row_key; // Row header
        for (int col : columns) {
            auto it = inner_map.find(col);
            if (it != inner_map.end()) {
                out_stream << std::setw(4) << it->second; // Value in the cell
            } else {
                out_stream << std::setw(4) << 0; // Default to 0 if no value exists
            }
        }
        out_stream << std::endl;
    }
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
        // int task_id = task_descriptor["task"];
        // int task_idx = task_id - 1; // ids are offset by 1 in the JSON file

        Task task_object;
        task_object.id              = task_descriptor["task"];
        task_object.processing_time = task_descriptor["processing_time"];
        task_object.machines        = std::set<int>();
        task_object.operators       = std::set<int>();
        
        for (auto& machine_descr : task_descriptor["machines"]) {
            int machine_id          = static_cast<int>(machine_descr["machine"]);
            int machine_idx         = machine_id - 1; // ids are offset by 1 in the JSON file
            
            std::set<int> partial_ops = {};
            for (auto& operator_id : machine_descr["operators"]) {
                int operator_idx = static_cast<int>(operator_id) - 1; // ids are offset by 1 in the JSON file
                partial_ops.insert(operator_idx);    // add the operator to the set of possible operators for that task
                task_object.operators.insert(operator_idx); // add the operator to the set of all possible operators for the task
            }
            task_object.compatibility.emplace(machine_idx, partial_ops);
            task_object.machines.emplace(machine_idx); // add the machine to the set of possible machines
        }

        // Commit log task details
        out_stream << "=== Imported task T" << task_object.id << " ===" << std::endl;
        out_stream << "* processing time: " << task_object.processing_time << std::endl;
        out_stream << "* possible supports: " << std::endl;
        
        for (auto& m : task_object.machines) {
            out_stream << "  M" << m + 1 << " with O";
            print_set(task_object.compatibility[m], 1, out_stream);
            out_stream << std::endl;
        }
        out_stream << std::endl;

        // Commit task object to instance
        inst_object.tasks.push_back(task_object);
    }

    out_stream << std::endl;
    // Iterate over jobs, import and print details
    for (auto& job_descriptor : inst_descriptor["jobs"]) {
        int job_id = job_descriptor["job"];
        int job_idx = job_id - 1; // ids are offset by 1 in the JSON file

        Job job_object;
        job_object.id = job_id;
        job_object.release_date = job_descriptor["release_date"];
        job_object.due_date     = job_descriptor["due_date"];
        job_object.weight       = job_descriptor["weight"];
        job_object.sequence     = std::vector<int>();

        for (int task_id : job_descriptor["sequence"]) {
            int task_idx = task_id - 1; // ids are offset by 1 in the JSON file
            job_object.sequence.push_back(task_idx); // ids are offset by 1 in the JSON file
            inst_object.tasks[task_idx].job_parent = job_idx;
        }
        
        // Commit log job details
        out_stream << "=== Imported job J" << job_object.id << " ===" << std::endl;
        out_stream << "* release date: " << job_object.release_date << std::endl;
        out_stream << "* due date: " << job_object.due_date << std::endl;
        out_stream << "* weight: " << job_object.weight << std::endl;
        out_stream << "* task sequence: ";
        print_sequence(job_object.sequence, 1, out_stream);
        out_stream << std::endl << std::endl;

        // Add job object to instance
        inst_object.jobs.push_back(job_object);
    }



    return inst_object;
}


bool all_stacks_are_empty(const std::map<int, std::deque<int>>& stacks) {
    for (const auto& pair : stacks) {
        if (!pair.second.empty()) {
            return false;
        }
    }
    return true;
}