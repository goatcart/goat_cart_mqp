#ifndef __SETTINGS_H
#define __SETTINGS_H

#include <yaml-cpp/yaml.h>

class Settings
{
private:
    YAML::Node config;
    Settings()
    {
        this->config = YAML::LoadFile("params.yml");
    }
public:
    static YAML::Node& get()
    {
        static Settings inst;
        return inst.config;
    }
};

#endif
