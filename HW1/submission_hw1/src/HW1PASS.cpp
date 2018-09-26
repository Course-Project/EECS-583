#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"

#include <iostream>
#include <fstream>

using namespace llvm;

#define DEBUG_TYPE "hw1"

namespace {
    // HW1 - The first implementation, without getAnalysisUsage.
    struct HW1 : public FunctionPass {
        static char ID; // Pass identification, replacement for typeid

        std::ofstream opcstats; // Output file

        HW1() : FunctionPass(ID) {
            opcstats.open("output.opcstats");
        }

        ~HW1() {
            opcstats.close();
        }

        bool runOnFunction(Function &F) override {
            if (F.isDeclaration()) return false;

            BranchProbabilityInfo *BPI = &getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();
            BlockFrequencyInfo *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();

            double totalInstructions = 0;
            double totalBiasedBranchInstructions = 0;
            double totalUnbiasedBranchInstructions = 0;
            double totalIntegerALUInstructions = 0;
            double totalFloatingPointALUInstructions = 0;
            double totalMemoryInstructions = 0;
            double totalOtherInstructions = 0;
            for (Function::iterator b = F.begin(); b != F.end(); b++) {
                BasicBlock *bb = &(*b);
                double numBranchInstructions = 0;

                for (BasicBlock::iterator i_iter = bb->begin(); i_iter != bb->end(); ++i_iter) {
                    Instruction *instr = &(*i_iter);

                    totalInstructions += BFI->getBlockProfileCount(bb).getValue();

		            switch (instr->getOpcode()) {
                        // Branch
                        case Instruction::Br:
                        case Instruction::Switch:
                        case Instruction::IndirectBr:
                            numBranchInstructions += BFI->getBlockProfileCount(bb).getValue();
                            break;
                        // Integer ALU
                        case Instruction::Add:
                        case Instruction::Sub:
                        case Instruction::Mul:
                        case Instruction::UDiv:
                        case Instruction::SDiv:
                        case Instruction::URem:
                        case Instruction::Shl:
                        case Instruction::LShr:
                        case Instruction::AShr:
                        case Instruction::And:
                        case Instruction::Or:
                        case Instruction::Xor:
                        case Instruction::ICmp:
                        case Instruction::SRem:
                            totalIntegerALUInstructions += BFI->getBlockProfileCount(bb).getValue();
                            break;
                        // Floating-point ALU
                        case Instruction::FAdd:
                        case Instruction::FSub:
                        case Instruction::FMul:
                        case Instruction::FDiv:
                        case Instruction::FRem:
                        case Instruction::FCmp:
                            totalFloatingPointALUInstructions += BFI->getBlockProfileCount(bb).getValue();
                            break;
                        // Memory
                        case Instruction::Alloca:
                        case Instruction::Load:
                        case Instruction::Store:
                        case Instruction::GetElementPtr:
                        case Instruction::Fence:
                        case Instruction::AtomicCmpXchg:
                        case Instruction::AtomicRMW:
                            totalMemoryInstructions += BFI->getBlockProfileCount(bb).getValue();
                            break;
                        // Others
                        default:
                            totalOtherInstructions += BFI->getBlockProfileCount(bb).getValue();
                            break;

                    }
                }

                totalUnbiasedBranchInstructions += numBranchInstructions;
                if (numBranchInstructions == 0) continue;

                for (auto d = succ_begin(bb); d != succ_end(bb); ++d) {
                    auto prob = BPI->getEdgeProbability(bb, *d).scale(100);

                    if (prob > 80) {
                        totalBiasedBranchInstructions += numBranchInstructions;
                        totalUnbiasedBranchInstructions -= numBranchInstructions;
                        break;
                    }

                }
            }
            
            errs().write_escaped(F.getName()) << ": Branch Instructions are executed " << totalBiasedBranchInstructions + totalUnbiasedBranchInstructions << " = " << "(" << totalBiasedBranchInstructions << " + " << totalUnbiasedBranchInstructions <<  ")" << " time.\n";
            errs().write_escaped(F.getName()) << ": Integer ALU Instructions are executed " << totalIntegerALUInstructions << " time.\n";
            errs().write_escaped(F.getName()) << ": Floating-point ALU Instructions are executed " << totalFloatingPointALUInstructions << " time.\n";
            errs().write_escaped(F.getName()) << ": Memory Instructions are executed " << totalMemoryInstructions << " time.\n";
            errs().write_escaped(F.getName()) << ": Other Instructions are executed " << totalOtherInstructions << " time.\n";


            opcstats << F.getName().data() << ", "
                << (totalInstructions == 0 ? totalInstructions++ : totalInstructions) << ", "
                << (totalIntegerALUInstructions / totalInstructions) << ", "
                << (totalFloatingPointALUInstructions / totalInstructions) << ", "
                << (totalMemoryInstructions / totalInstructions) << ", "
                << (totalBiasedBranchInstructions / totalInstructions) << ", "
                << (totalUnbiasedBranchInstructions / totalInstructions) << ", "
                << (totalOtherInstructions / totalInstructions) << std::endl;
            return false;
        }

        void getAnalysisUsage(AnalysisUsage &AU) const {
            AU.addRequired<BranchProbabilityInfoWrapperPass>();
            AU.addRequired<BlockFrequencyInfoWrapperPass>();
        }
    };

}

char HW1::ID = 0;
static RegisterPass<HW1> X("hw1", "HW1 Pass");
