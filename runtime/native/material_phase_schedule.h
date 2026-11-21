#pragma once
#include <limits>

// Host-side schedule recognition only. It does not skip or modify a solver
// update. Native trace replay must validate this ordering before intervention.
namespace deformable {
enum class MaterialPhase { Ordinary, Terminal, Rejected };
struct MaterialPhaseSchedule {
    unsigned positions=0, partitions=0, advances=0, merges=0, next_partition=0;
    bool closed=true, failed=false;
    bool reject() { failed=true; return false; }
    bool begin(unsigned n, unsigned p) {
        if (failed || !closed || n==0 || p==0 || n==std::numeric_limits<unsigned>::max()) return reject();
        positions=n; partitions=p; advances=merges=next_partition=0; closed=false;
        return true;
    }
    bool advance() {
        if (failed || closed || next_partition || advances>=positions || merges!=advances) return reject();
        ++advances; return true;
    }
    MaterialPhase phase() const {
        if (failed || closed || merges>positions) return MaterialPhase::Rejected;
        const unsigned expected=merges<positions?merges+1:positions;
        if (advances!=expected) return MaterialPhase::Rejected;
        return merges==positions?MaterialPhase::Terminal:MaterialPhase::Ordinary;
    }
    MaterialPhase partition(unsigned index) {
        const auto result=phase();
        if (result==MaterialPhase::Rejected || index!=next_partition || index>=partitions) {
            reject(); return MaterialPhase::Rejected;
        }
        ++next_partition; return result;
    }
    bool merge() {
        if (phase()==MaterialPhase::Rejected || next_partition!=partitions) return reject();
        next_partition=0; ++merges; return true;
    }
    bool finish() {
        if (failed || closed || advances!=positions || merges!=positions+1 || next_partition) return reject();
        closed=true; return true;
    }
};
}
