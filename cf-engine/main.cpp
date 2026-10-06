#include "adapter.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int cf_probe_main(int, char **);

int main(int argc, char ** argv) {
    std::vector<char *> forwarded{argv[0]};
    std::string model;
    bool engine_given = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        const char * env = nullptr;
        if (arg == "--engine") { env = "CF_MODE"; engine_given = true; }
        else if (arg == "--cf-plan") env = "CF_PLAN";
        else if (arg == "--cf-kernel") env = "CF_KERNEL";
        else if (arg == "--cf-report") env = "CF_REPORT";
        else if (arg == "--cf-memory") env = "CF_MEMORY";
        else if (arg == "--cf-logits") env = "CF_LOGITS";
        else if (arg == "--cf-routes") env = "CF_ROUTES";
        else if (arg == "--cf-schedule") env = "CF_SCHEDULE";
        else if (arg == "--cf-scheduler") env = "CF_SCHEDULER";
        else if (arg == "--cf-capture-dir") env = "CF_CAPTURE_DIR";
        else if (arg == "--cf-validate") { setenv("CF_VALIDATE", "1", 1); continue; }
        if (env) {
            if (++i >= argc) { std::cerr << "missing value for " << arg << '\n'; return 2; }
            setenv(env, argv[i], 1);
        } else {
            forwarded.push_back(argv[i]);
            if ((arg == "-m" || arg == "--model") && i + 1 < argc) model = argv[i + 1];
        }
    }
    if (!engine_given) { std::cerr << "CF requires --engine reference|cf\n"; return 2; }
    if (cf_initialize(model.c_str())) return 3;
    forwarded.push_back(nullptr);
    int rc = 0;
    try { rc = cf_probe_main(int(forwarded.size()) - 1, forwarded.data()); }
    catch (const std::exception & e) { std::cerr << "CF session: " << e.what() << '\n'; rc = 3; }
    int validation = cf_finish();
    return rc ? rc : validation;
}
