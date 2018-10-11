//===Freuqent Path Loop Invariant Code Motion Pass ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// EECS583 F18 - This pass can be used as a template for your Frequent Path LICM
//               homework assignment. The pass gets registered as "fplicm".
//
// This pass performs loop invariant code motion, attempting to remove as much
// code from the body of a loop as possible.  It does this by either hoisting
// code into the preheader block, or by sinking code to the exit blocks if it is
// safe.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <algorithm>
#include <utility>

#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"

#include <vector>
#include <map>

using namespace llvm;

#define DEBUG_TYPE "fplicm"

STATISTIC(NumSunk, "Number of instructions sunk out of loop");
STATISTIC(NumHoisted, "Number of instructions hoisted out of loop");
STATISTIC(NumMovedLoads, "Number of load insts hoisted or sunk");
STATISTIC(NumMovedCalls, "Number of call insts hoisted or sunk");
STATISTIC(NumPromoted, "Number of memory locations promoted to registers");

/// Memory promotion is enabled by default.
static cl::opt<bool>
    DisablePromotion("disable-fplicm-promotion", cl::Hidden, cl::init(false),
                     cl::desc("Disable memory promotion in FPLICM pass"));

static cl::opt<uint32_t> MaxNumUsesTraversed(
    "fplicm-max-num-uses-traversed", cl::Hidden, cl::init(8),
    cl::desc("Max num uses visited for identifying load "
             "invariance in loop using invariant start (default = 8)"));

static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI);
static bool isNotUsedOrFreeInLoop(const Instruction &I, const Loop *CurLoop,
                                  const LoopSafetyInfo *SafetyInfo,
                                  TargetTransformInfo *TTI, bool &FreeInLoop);
static bool hoist(Instruction &I, const DominatorTree *DT, const Loop *CurLoop,
                  const LoopSafetyInfo *SafetyInfo,
                  OptimizationRemarkEmitter *ORE);
static bool sink(Instruction &I, LoopInfo *LI, DominatorTree *DT,
                 const Loop *CurLoop, LoopSafetyInfo *SafetyInfo,
                 OptimizationRemarkEmitter *ORE, bool FreeInLoop);
static bool isSafeToExecuteUnconditionally(Instruction &Inst,
                                           const DominatorTree *DT,
                                           const Loop *CurLoop,
                                           const LoopSafetyInfo *SafetyInfo,
                                           OptimizationRemarkEmitter *ORE,
                                           const Instruction *CtxI = nullptr);
static bool pointerInvalidatedByLoop(Value *V, uint64_t Size,
                                     const AAMDNodes &AAInfo,
                                     AliasSetTracker *CurAST);
static Instruction *
CloneInstructionInExitBlock(Instruction &I, BasicBlock &ExitBlock, PHINode &PN,
                            const LoopInfo *LI,
                            const LoopSafetyInfo *SafetyInfo);

namespace {
struct LoopInvariantCodeMotion {
  bool runOnLoop(Loop *L, AliasAnalysis *AA, LoopInfo *LI, DominatorTree *DT,
                 TargetLibraryInfo *TLI, TargetTransformInfo *TTI,
                 ScalarEvolution *SE, MemorySSA *MSSA,
                 OptimizationRemarkEmitter *ORE, bool DeleteAST,
                 BranchProbabilityInfo *BPI, BlockFrequencyInfo *BFI);

  DenseMap<Loop *, AliasSetTracker *> &getLoopToAliasSetMap() {
    return LoopToAliasSetMap;
  }

private:
  DenseMap<Loop *, AliasSetTracker *> LoopToAliasSetMap;

  AliasSetTracker *collectAliasInfoForLoop(Loop *L, LoopInfo *LI,
                                           AliasAnalysis *AA);

  std::vector<BasicBlock *> FrequentLoopPath;
};

struct FPLICMPass : public LoopPass {
  static char ID; // Pass identification, replacement for typeid
  FPLICMPass() : LoopPass(ID) {}

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    auto *SE = getAnalysisIfAvailable<ScalarEvolutionWrapperPass>();
    MemorySSA *MSSA = EnableMSSALoopDependency
                          ? (&getAnalysis<MemorySSAWrapperPass>().getMSSA())
                          : nullptr;
    // For the old PM, we can't use OptimizationRemarkEmitter as an analysis
    // pass.  Function analyses need to be preserved across loop transformations
    // but ORE cannot be preserved (see comment before the pass definition).
    OptimizationRemarkEmitter ORE(L->getHeader()->getParent());
    return LICM.runOnLoop(L,
                          &getAnalysis<AAResultsWrapperPass>().getAAResults(),
                          &getAnalysis<LoopInfoWrapperPass>().getLoopInfo(),
                          &getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
                          &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(),
                          &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(
                              *L->getHeader()->getParent()),
                          SE ? &SE->getSE() : nullptr, MSSA, &ORE, false,
                          &getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI(),
                          &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI());
  }

  /// This transformation requires natural loop information & requires that
  /// loop preheaders be inserted into the CFG...
  ///
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.addRequired<BlockFrequencyInfoWrapperPass>();

    AU.addRequired<TargetLibraryInfoWrapperPass>();
    if (EnableMSSALoopDependency)
      AU.addRequired<MemorySSAWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    getLoopAnalysisUsage(AU);
  }

  using llvm::Pass::doFinalization;

  bool doFinalization() override {
    assert(LICM.getLoopToAliasSetMap().empty() &&
           "Didn't free loop alias sets");
    return false;
  }

private:
  LoopInvariantCodeMotion LICM;

  /// cloneBasicBlockAnalysis - Simple Analysis hook. Clone alias set info.
  void cloneBasicBlockAnalysis(BasicBlock *From, BasicBlock *To,
                               Loop *L) override;

  /// deleteAnalysisValue - Simple Analysis hook. Delete value V from alias
  /// set.
  void deleteAnalysisValue(Value *V, Loop *L) override;

  /// Simple Analysis hook. Delete loop L from alias set map.
  void deleteAnalysisLoop(Loop *L) override;
};

} // namespace

char FPLICMPass::ID = 0;
static RegisterPass<FPLICMPass> X("fplicm", "Loop Invariant Code Motion");

bool LoopInvariantCodeMotion::runOnLoop(
    Loop *L, AliasAnalysis *AA, LoopInfo *LI, DominatorTree *DT,
    TargetLibraryInfo *TLI, TargetTransformInfo *TTI, ScalarEvolution *SE,
    MemorySSA *MSSA, OptimizationRemarkEmitter *ORE, bool DeleteAST,
    BranchProbabilityInfo *BPI, BlockFrequencyInfo *BFI) {
  bool Changed = false;
  assert(L->isLCSSAForm(*DT) && "Loop is not in LCSSA form.");

  // Get the preheader block to move instructions into...
  BasicBlock *Preheader = L->getLoopPreheader();

  // ========= 1
  // Get the begining basic block
  BasicBlock *LoopHeader = L->getHeader();

  // Start from the loop header
  BasicBlock *BB = LoopHeader;
  FrequentLoopPath.push_back(BB);

  do {
    // Choose the most frequent successor
    auto MaxProb = BranchProbability::getZero();
    BasicBlock *MaxSucc = nullptr;
    for (auto SI = succ_begin(BB); SI != succ_end(BB); ++SI) {
      BasicBlock *BB_succ = *SI;
      auto Prob = BPI->getEdgeProbability(BB, BB_succ);
      if (Prob > MaxProb) {
        MaxProb = Prob;
        MaxSucc = BB_succ;
      }
    }
    BB = MaxSucc;

    // Add it to the vector
    FrequentLoopPath.push_back(BB);
  } while(!(L->isLoopExiting(BB) || L->isLoopLatch(BB)));

  std::vector<StoreInst *> StoreInstructions;
  for (BasicBlock *BB : FrequentLoopPath) {
    for (Instruction &I : *BB) {
      if (I.getOpcode() == Instruction::Store) {
        // Find a store instruction
        StoreInstructions.push_back(dyn_cast<StoreInst>(&I));
      }
    }
  }

  // Process frequent path
  std::vector<Instruction *> HoistInstructions;
  for (BasicBlock *BB : FrequentLoopPath) {
    // errs() << "[h1994st] Basic Block ";
    // BB->printAsOperand(errs(), false);
    // errs() << "\n";

    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        // A load instruction

        bool canHoist = true;
        // Check whether src operands are modified/defined in this path
        auto *Src = cast<Instruction>(LI->getOperand(0));
        if (std::find(
          FrequentLoopPath.begin(), FrequentLoopPath.end(),
          Src->getParent()) != FrequentLoopPath.end()) {
          canHoist = false;
        }

        // Check whether there exists any store instruction modify
        // the same address
        for (StoreInst *SI : StoreInstructions) {
          if (SI->getOperand(1) == Src) {
            canHoist = false;
            break;
          }
        }

        if (canHoist) {
          // errs() << "[h1994st] Can hoist: " << *LI << "\n";
          // errs() << "[h1994st] Load " << *LI << "\n";
          // errs() << LI->getOperand(0) << ":" << *(LI->getOperand(0)) << "\n";

          HoistInstructions.push_back(&I);
        }
      }
    }
  }
  // ========= 1

  // ========= 2. Hoist & Fix
  std::map<Instruction *, AllocaInst *> SrcToVal;
  for (Instruction *I : HoistInstructions) {
    // Hoist a instruction
    errs() << "[h1994st] Hoist: " << *I << "\n";
    
    auto *Src = cast<Instruction>(I->getOperand(0));

    // Create a temp variable
    AllocaInst *Val;
    if (SrcToVal[Src] == nullptr) {
      Val = new AllocaInst(I->getType(), 0, "", Preheader->getTerminator());
      errs() << "[h1994st] Create new stack variable: " << *Val << "\n";
      SrcToVal[Src] = Val;
    } else {
      Val = SrcToVal[Src];
      errs() << "[h1994st] Use existing stack variable: " << *Val << "\n";
    }

    // Clone a load instruction, and hoist it
    Instruction *ExtraLI = I->clone();
    ExtraLI->insertBefore(Preheader->getTerminator());
    Changed |= true;

    // Add one more store instruction
    StoreInst *ExtraSI = new StoreInst(ExtraLI, Val, Preheader->getTerminator());

    // Change the operand of the load instruction
    I->setOperand(0, Val);

    // Fix it
    for (BasicBlock *BB : L->blocks()) {
      if (std::find(
        FrequentLoopPath.begin(), FrequentLoopPath.end(),
        BB) != FrequentLoopPath.end()) {
        continue;
      }

      // Infrequent BB
      for (Instruction &IFI : *BB) {
        if (auto *SI = dyn_cast<StoreInst>(&IFI)) {
          if (SI->getOperand(1) == Src) {
            errs() << "[h1994st] Fix: " << *SI << "\n";

            Instruction *NewI = ExtraLI->clone();
            // LoadInst *NewI = new LoadInst(Src, "fix", BB);
            // NewI->setName(I->getName());
            NewI->insertBefore(BB->getTerminator());
            StoreInst *NewSI = new StoreInst(NewI, Val, BB->getTerminator());
            errs() << "[h1994st] Insert new instruction: " << *NewI << "\n";
            errs() << "[h1994st] Insert new instruction: " << *NewSI << "\n";
            errs() << "[h1994st] Done!\n";
            break;
          }
        }
      }
    }
  }
  // ========= 2

  AliasSetTracker *CurAST = collectAliasInfoForLoop(L, LI, AA);

  // Compute loop safety information.
  LoopSafetyInfo SafetyInfo;
  computeLoopSafetyInfo(&SafetyInfo, L);

  // We want to visit all of the instructions in this loop... that are not parts
  // of our subloops (they have already had their invariants hoisted out of
  // their loop, into this loop, so there is no need to process the BODIES of
  // the subloops).
  //
  // Traverse the body of the loop in depth first order on the dominator tree so
  // that we are guaranteed to see definitions before we see uses.  This allows
  // us to sink instructions in one pass, without iteration.  After sinking
  // instructions, we perform another pass to hoist them out of the loop.
  //
  if (L->hasDedicatedExits())
    Changed |= sinkRegion(DT->getNode(L->getHeader()), AA, LI, DT, TLI, TTI, L,
                          CurAST, &SafetyInfo, ORE);
  if (Preheader)
    Changed |= hoistRegion(DT->getNode(L->getHeader()), AA, LI, DT, TLI, L,
                           CurAST, &SafetyInfo, ORE);

  // Check that neither this loop nor its parent have had LCSSA broken. LICM is
  // specifically moving instructions across the loop boundary and so it is
  // especially in need of sanity checking here.
  assert(L->isLCSSAForm(*DT) && "Loop not left in LCSSA form after LICM!");
  assert((!L->getParentLoop() || L->getParentLoop()->isLCSSAForm(*DT)) &&
         "Parent loop not left in LCSSA form after LICM!");

  // If this loop is nested inside of another one, save the alias information
  // for when we process the outer loop.
  if (L->getParentLoop() && !DeleteAST)
    LoopToAliasSetMap[L] = CurAST;
  else
    delete CurAST;

  if (Changed && SE)
    SE->forgetLoopDispositions(L);
  return Changed;
}

/// Walk the specified region of the CFG (defined by all blocks dominated by
/// the specified block, and that are in the current loop) in reverse depth
/// first order w.r.t the DominatorTree.  This allows us to visit uses before
/// definitions, allowing us to sink a loop body in one pass without iteration.
///
bool llvm::sinkRegion(DomTreeNode *N, AliasAnalysis *AA, LoopInfo *LI,
                      DominatorTree *DT, TargetLibraryInfo *TLI,
                      TargetTransformInfo *TTI, Loop *CurLoop,
                      AliasSetTracker *CurAST, LoopSafetyInfo *SafetyInfo,
                      OptimizationRemarkEmitter *ORE) {

  // Verify inputs.
  assert(N != nullptr && AA != nullptr && LI != nullptr && DT != nullptr &&
         CurLoop != nullptr && CurAST != nullptr && SafetyInfo != nullptr &&
         "Unexpected input to sinkRegion");

  // We want to visit children before parents. We will enque all the parents
  // before their children in the worklist and process the worklist in reverse
  // order.
  SmallVector<DomTreeNode *, 16> Worklist = collectChildrenInLoop(N, CurLoop);

  bool Changed = false;
  for (DomTreeNode *DTN : reverse(Worklist)) {
    BasicBlock *BB = DTN->getBlock();
    // Only need to process the contents of this block if it is not part of a
    // subloop (which would already have been processed).
    if (inSubLoop(BB, CurLoop, LI))
      continue;

    for (BasicBlock::iterator II = BB->end(); II != BB->begin();) {
      Instruction &I = *--II;

      // If the instruction is dead, we would try to sink it because it isn't
      // used in the loop, instead, just delete it.
      if (isInstructionTriviallyDead(&I, TLI)) {
        DEBUG(dbgs() << "LICM deleting dead inst: " << I << '\n');
        ++II;
        CurAST->deleteValue(&I);
        I.eraseFromParent();
        Changed = true;
        continue;
      }

      // Check to see if we can sink this instruction to the exit blocks
      // of the loop.  We can do this if the all users of the instruction are
      // outside of the loop.  In this case, it doesn't even matter if the
      // operands of the instruction are loop invariant.
      //
      bool FreeInLoop = false;
      if (isNotUsedOrFreeInLoop(I, CurLoop, SafetyInfo, TTI, FreeInLoop) &&
          canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo, ORE)) {
        if (sink(I, LI, DT, CurLoop, SafetyInfo, ORE, FreeInLoop)) {
          if (!FreeInLoop) {
            ++II;
            CurAST->deleteValue(&I);
            I.eraseFromParent();
          }
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

/// Walk the specified region of the CFG (defined by all blocks dominated by
/// the specified block, and that are in the current loop) in depth first
/// order w.r.t the DominatorTree.  This allows us to visit definitions before
/// uses, allowing us to hoist a loop body in one pass without iteration.
///
bool llvm::hoistRegion(DomTreeNode *N, AliasAnalysis *AA, LoopInfo *LI,
                       DominatorTree *DT, TargetLibraryInfo *TLI, Loop *CurLoop,
                       AliasSetTracker *CurAST, LoopSafetyInfo *SafetyInfo,
                       OptimizationRemarkEmitter *ORE) {
  // Verify inputs.
  assert(N != nullptr && AA != nullptr && LI != nullptr && DT != nullptr &&
         CurLoop != nullptr && CurAST != nullptr && SafetyInfo != nullptr &&
         "Unexpected input to hoistRegion");

  // We want to visit parents before children. We will enque all the parents
  // before their children in the worklist and process the worklist in order.
  SmallVector<DomTreeNode *, 16> Worklist = collectChildrenInLoop(N, CurLoop);

  bool Changed = false;
  for (DomTreeNode *DTN : Worklist) {
    BasicBlock *BB = DTN->getBlock();
    // Only need to process the contents of this block if it is not part of a
    // subloop (which would already have been processed).
    if (!inSubLoop(BB, CurLoop, LI))
      for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;) {
        Instruction &I = *II++;
        // Try constant folding this instruction.  If all the operands are
        // constants, it is technically hoistable, but it would be better to
        // just fold it.
        if (Constant *C = ConstantFoldInstruction(
                &I, I.getModule()->getDataLayout(), TLI)) {
          DEBUG(dbgs() << "LICM folding inst: " << I << "  --> " << *C << '\n');
          CurAST->copyValue(&I, C);
          I.replaceAllUsesWith(C);
          if (isInstructionTriviallyDead(&I, TLI)) {
            CurAST->deleteValue(&I);
            I.eraseFromParent();
          }
          Changed = true;
          continue;
        }

        // Attempt to remove floating point division out of the loop by
        // converting it to a reciprocal multiplication.
        if (I.getOpcode() == Instruction::FDiv &&
            CurLoop->isLoopInvariant(I.getOperand(1)) &&
            I.hasAllowReciprocal()) {
          auto Divisor = I.getOperand(1);
          auto One = llvm::ConstantFP::get(Divisor->getType(), 1.0);
          auto ReciprocalDivisor = BinaryOperator::CreateFDiv(One, Divisor);
          ReciprocalDivisor->setFastMathFlags(I.getFastMathFlags());
          ReciprocalDivisor->insertBefore(&I);

          auto Product =
              BinaryOperator::CreateFMul(I.getOperand(0), ReciprocalDivisor);
          Product->setFastMathFlags(I.getFastMathFlags());
          Product->insertAfter(&I);
          I.replaceAllUsesWith(Product);
          I.eraseFromParent();

          hoist(*ReciprocalDivisor, DT, CurLoop, SafetyInfo, ORE);
          Changed = true;
          continue;
        }

        // Try hoisting the instruction out to the preheader.  We can only do
        // this if all of the operands of the instruction are loop invariant and
        // if it is safe to hoist the instruction.
        //
        if (CurLoop->hasLoopInvariantOperands(&I) &&
            canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo, ORE) &&
            isSafeToExecuteUnconditionally(
                I, DT, CurLoop, SafetyInfo, ORE,
                CurLoop->getLoopPreheader()->getTerminator()))
          Changed |= hoist(I, DT, CurLoop, SafetyInfo, ORE);
      }
  }

  return Changed;
}

/// Computes loop safety information, checks loop body & header
/// for the possibility of may throw exception.
///
void llvm::computeLoopSafetyInfo(LoopSafetyInfo *SafetyInfo, Loop *CurLoop) {
  assert(CurLoop != nullptr && "CurLoop cant be null");
  BasicBlock *Header = CurLoop->getHeader();
  // Setting default safety values.
  SafetyInfo->MayThrow = false;
  SafetyInfo->HeaderMayThrow = false;
  // Iterate over header and compute safety info.
  for (BasicBlock::iterator I = Header->begin(), E = Header->end();
       (I != E) && !SafetyInfo->HeaderMayThrow; ++I)
    SafetyInfo->HeaderMayThrow |=
        !isGuaranteedToTransferExecutionToSuccessor(&*I);

  SafetyInfo->MayThrow = SafetyInfo->HeaderMayThrow;
  // Iterate over loop instructions and compute safety info.
  // Skip header as it has been computed and stored in HeaderMayThrow.
  // The first block in loopinfo.Blocks is guaranteed to be the header.
  assert(Header == *CurLoop->getBlocks().begin() &&
         "First block must be header");
  for (Loop::block_iterator BB = std::next(CurLoop->block_begin()),
                            BBE = CurLoop->block_end();
       (BB != BBE) && !SafetyInfo->MayThrow; ++BB)
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end();
         (I != E) && !SafetyInfo->MayThrow; ++I)
      SafetyInfo->MayThrow |= !isGuaranteedToTransferExecutionToSuccessor(&*I);

  // Compute funclet colors if we might sink/hoist in a function with a funclet
  // personality routine.
  Function *Fn = CurLoop->getHeader()->getParent();
  if (Fn->hasPersonalityFn())
    if (Constant *PersonalityFn = Fn->getPersonalityFn())
      if (isFuncletEHPersonality(classifyEHPersonality(PersonalityFn)))
        SafetyInfo->BlockColors = colorEHFunclets(*Fn);
}

// Return true if LI is invariant within scope of the loop. LI is invariant if
// CurLoop is dominated by an invariant.start representing the same memory
// location and size as the memory location LI loads from, and also the
// invariant.start has no uses.
static bool isLoadInvariantInLoop(LoadInst *LI, DominatorTree *DT,
                                  Loop *CurLoop) {
  Value *Addr = LI->getOperand(0);
  const DataLayout &DL = LI->getModule()->getDataLayout();
  const uint32_t LocSizeInBits = DL.getTypeSizeInBits(
      cast<PointerType>(Addr->getType())->getElementType());

  // if the type is i8 addrspace(x)*, we know this is the type of
  // llvm.invariant.start operand
  auto *PtrInt8Ty = PointerType::get(Type::getInt8Ty(LI->getContext()),
                                     LI->getPointerAddressSpace());
  unsigned BitcastsVisited = 0;
  // Look through bitcasts until we reach the i8* type (this is invariant.start
  // operand type).
  while (Addr->getType() != PtrInt8Ty) {
    auto *BC = dyn_cast<BitCastInst>(Addr);
    // Avoid traversing high number of bitcast uses.
    if (++BitcastsVisited > MaxNumUsesTraversed || !BC)
      return false;
    Addr = BC->getOperand(0);
  }

  unsigned UsesVisited = 0;
  // Traverse all uses of the load operand value, to see if invariant.start is
  // one of the uses, and whether it dominates the load instruction.
  for (auto *U : Addr->users()) {
    // Avoid traversing for Load operand with high number of users.
    if (++UsesVisited > MaxNumUsesTraversed)
      return false;
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
    // If there are escaping uses of invariant.start instruction, the load maybe
    // non-invariant.
    if (!II || II->getIntrinsicID() != Intrinsic::invariant_start ||
        !II->use_empty())
      continue;
    unsigned InvariantSizeInBits =
        cast<ConstantInt>(II->getArgOperand(0))->getSExtValue() * 8;
    // Confirm the invariant.start location size contains the load operand size
    // in bits. Also, the invariant.start should dominate the load, and we
    // should not hoist the load out of a loop that contains this dominating
    // invariant.start.
    if (LocSizeInBits <= InvariantSizeInBits &&
        DT->properlyDominates(II->getParent(), CurLoop->getHeader()))
      return true;
  }

  return false;
}

bool llvm::canSinkOrHoistInst(Instruction &I, AAResults *AA, DominatorTree *DT,
                              Loop *CurLoop, AliasSetTracker *CurAST,
                              LoopSafetyInfo *SafetyInfo,
                              OptimizationRemarkEmitter *ORE) {
  // SafetyInfo is nullptr if we are checking for sinking from preheader to
  // loop body.
  const bool SinkingToLoopBody = !SafetyInfo;
  // Loads have extra constraints we have to verify before we can hoist them.
  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    if (!LI->isUnordered())
      return false; // Don't sink/hoist volatile or ordered atomic loads!

    // Loads from constant memory are always safe to move, even if they end up
    // in the same alias set as something that ends up being modified.
    if (AA->pointsToConstantMemory(LI->getOperand(0)))
      return true;
    if (LI->getMetadata(LLVMContext::MD_invariant_load))
      return true;

    if (LI->isAtomic() && SinkingToLoopBody)
      return false; // Don't sink unordered atomic loads to loop body.

    // This checks for an invariant.start dominating the load.
    if (isLoadInvariantInLoop(LI, DT, CurLoop))
      return true;

    // Don't hoist loads which have may-aliased stores in loop.
    uint64_t Size = 0;
    if (LI->getType()->isSized())
      Size = I.getModule()->getDataLayout().getTypeStoreSize(LI->getType());

    AAMDNodes AAInfo;
    LI->getAAMetadata(AAInfo);

    bool Invalidated =
        pointerInvalidatedByLoop(LI->getOperand(0), Size, AAInfo, CurAST);
    // Check loop-invariant address because this may also be a sinkable load
    // whose address is not necessarily loop-invariant.
    if (ORE && Invalidated && CurLoop->isLoopInvariant(LI->getPointerOperand()))
      ORE->emit([&]() {
        return OptimizationRemarkMissed(
                   DEBUG_TYPE, "LoadWithLoopInvariantAddressInvalidated", LI)
               << "failed to move load with loop-invariant address "
                  "because the loop may invalidate its value";
      });

    return !Invalidated;
  } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
    // Don't sink or hoist dbg info; it's legal, but not useful.
    if (isa<DbgInfoIntrinsic>(I))
      return false;

    // Don't sink calls which can throw.
    if (CI->mayThrow())
      return false;

    // Handle simple cases by querying alias analysis.
    FunctionModRefBehavior Behavior = AA->getModRefBehavior(CI);
    if (Behavior == FMRB_DoesNotAccessMemory)
      return true;
    if (AliasAnalysis::onlyReadsMemory(Behavior)) {
      // A readonly argmemonly function only reads from memory pointed to by
      // it's arguments with arbitrary offsets.  If we can prove there are no
      // writes to this memory in the loop, we can hoist or sink.
      if (AliasAnalysis::onlyAccessesArgPointees(Behavior)) {
        for (Value *Op : CI->arg_operands())
          if (Op->getType()->isPointerTy() &&
              pointerInvalidatedByLoop(Op, MemoryLocation::UnknownSize,
                                       AAMDNodes(), CurAST))
            return false;
        return true;
      }
      // If this call only reads from memory and there are no writes to memory
      // in the loop, we can hoist or sink the call as appropriate.
      bool FoundMod = false;
      for (AliasSet &AS : *CurAST) {
        if (!AS.isForwardingAliasSet() && AS.isMod()) {
          FoundMod = true;
          break;
        }
      }
      if (!FoundMod)
        return true;
    }

    // FIXME: This should use mod/ref information to see if we can hoist or
    // sink the call.

    return false;
  }

  // Only these instructions are hoistable/sinkable.
  if (!isa<BinaryOperator>(I) && !isa<CastInst>(I) && !isa<SelectInst>(I) &&
      !isa<GetElementPtrInst>(I) && !isa<CmpInst>(I) &&
      !isa<InsertElementInst>(I) && !isa<ExtractElementInst>(I) &&
      !isa<ShuffleVectorInst>(I) && !isa<ExtractValueInst>(I) &&
      !isa<InsertValueInst>(I))
    return false;

  // If we are checking for sinking from preheader to loop body it will be
  // always safe as there is no speculative execution.
  if (SinkingToLoopBody)
    return true;

  // TODO: Plumb the context instruction through to make hoisting and sinking
  // more powerful. Hoisting of loads already works due to the special casing
  // above.
  return isSafeToExecuteUnconditionally(I, DT, CurLoop, SafetyInfo, nullptr);
}

/// Returns true if a PHINode is a trivially replaceable with an
/// Instruction.
/// This is true when all incoming values are that instruction.
/// This pattern occurs most often with LCSSA PHI nodes.
///
static bool isTriviallyReplacablePHI(const PHINode &PN, const Instruction &I) {
  for (const Value *IncValue : PN.incoming_values())
    if (IncValue != &I)
      return false;

  return true;
}

/// Return true if the instruction is free in the loop.
static bool isFreeInLoop(const Instruction &I, const Loop *CurLoop,
                         const TargetTransformInfo *TTI) {

  if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    if (TTI->getUserCost(GEP) != TargetTransformInfo::TCC_Free)
      return false;
    // For a GEP, we cannot simply use getUserCost because currently it
    // optimistically assume that a GEP will fold into addressing mode
    // regardless of its users.
    const BasicBlock *BB = GEP->getParent();
    for (const User *U : GEP->users()) {
      const Instruction *UI = cast<Instruction>(U);
      if (CurLoop->contains(UI) &&
          (BB != UI->getParent() ||
           (!isa<StoreInst>(UI) && !isa<LoadInst>(UI))))
        return false;
    }
    return true;
  } else
    return TTI->getUserCost(&I) == TargetTransformInfo::TCC_Free;
}

/// Return true if the only users of this instruction are outside of
/// the loop. If this is true, we can sink the instruction to the exit
/// blocks of the loop.
///
/// We also return true if the instruction could be folded away in lowering.
/// (e.g.,  a GEP can be folded into a load as an addressing mode in the loop).
static bool isNotUsedOrFreeInLoop(const Instruction &I, const Loop *CurLoop,
                                  const LoopSafetyInfo *SafetyInfo,
                                  TargetTransformInfo *TTI, bool &FreeInLoop) {
  const auto &BlockColors = SafetyInfo->BlockColors;
  bool IsFree = isFreeInLoop(I, CurLoop, TTI);
  for (const User *U : I.users()) {
    const Instruction *UI = cast<Instruction>(U);
    if (const PHINode *PN = dyn_cast<PHINode>(UI)) {
      const BasicBlock *BB = PN->getParent();
      // We cannot sink uses in catchswitches.
      if (isa<CatchSwitchInst>(BB->getTerminator()))
        return false;

      // We need to sink a callsite to a unique funclet.  Avoid sinking if the
      // phi use is too muddled.
      if (isa<CallInst>(I))
        if (!BlockColors.empty() &&
            BlockColors.find(const_cast<BasicBlock *>(BB))->second.size() != 1)
          return false;
    }

    if (CurLoop->contains(UI)) {
      if (IsFree) {
        FreeInLoop = true;
        continue;
      }
      return false;
    }
  }
  return true;
}

static Instruction *
CloneInstructionInExitBlock(Instruction &I, BasicBlock &ExitBlock, PHINode &PN,
                            const LoopInfo *LI,
                            const LoopSafetyInfo *SafetyInfo) {
  Instruction *New;
  if (auto *CI = dyn_cast<CallInst>(&I)) {
    const auto &BlockColors = SafetyInfo->BlockColors;

    // Sinking call-sites need to be handled differently from other
    // instructions.  The cloned call-site needs a funclet bundle operand
    // appropriate for it's location in the CFG.
    SmallVector<OperandBundleDef, 1> OpBundles;
    for (unsigned BundleIdx = 0, BundleEnd = CI->getNumOperandBundles();
         BundleIdx != BundleEnd; ++BundleIdx) {
      OperandBundleUse Bundle = CI->getOperandBundleAt(BundleIdx);
      if (Bundle.getTagID() == LLVMContext::OB_funclet)
        continue;

      OpBundles.emplace_back(Bundle);
    }

    if (!BlockColors.empty()) {
      const ColorVector &CV = BlockColors.find(&ExitBlock)->second;
      assert(CV.size() == 1 && "non-unique color for exit block!");
      BasicBlock *BBColor = CV.front();
      Instruction *EHPad = BBColor->getFirstNonPHI();
      if (EHPad->isEHPad())
        OpBundles.emplace_back("funclet", EHPad);
    }

    New = CallInst::Create(CI, OpBundles);
  } else {
    New = I.clone();
  }

  ExitBlock.getInstList().insert(ExitBlock.getFirstInsertionPt(), New);
  if (!I.getName().empty())
    New->setName(I.getName() + ".le");

  // Build LCSSA PHI nodes for any in-loop operands. Note that this is
  // particularly cheap because we can rip off the PHI node that we're
  // replacing for the number and blocks of the predecessors.
  // OPT: If this shows up in a profile, we can instead finish sinking all
  // invariant instructions, and then walk their operands to re-establish
  // LCSSA. That will eliminate creating PHI nodes just to nuke them when
  // sinking bottom-up.
  for (User::op_iterator OI = New->op_begin(), OE = New->op_end(); OI != OE;
       ++OI)
    if (Instruction *OInst = dyn_cast<Instruction>(*OI))
      if (Loop *OLoop = LI->getLoopFor(OInst->getParent()))
        if (!OLoop->contains(&PN)) {
          PHINode *OpPN =
              PHINode::Create(OInst->getType(), PN.getNumIncomingValues(),
                              OInst->getName() + ".lcssa", &ExitBlock.front());
          for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
            OpPN->addIncoming(OInst, PN.getIncomingBlock(i));
          *OI = OpPN;
        }
  return New;
}

static Instruction *sinkThroughTriviallyReplacablePHI(
    PHINode *TPN, Instruction *I, LoopInfo *LI,
    SmallDenseMap<BasicBlock *, Instruction *, 32> &SunkCopies,
    const LoopSafetyInfo *SafetyInfo, const Loop *CurLoop) {
  assert(isTriviallyReplacablePHI(*TPN, *I) &&
         "Expect only trivially replacalbe PHI");
  BasicBlock *ExitBlock = TPN->getParent();
  Instruction *New;
  auto It = SunkCopies.find(ExitBlock);
  if (It != SunkCopies.end())
    New = It->second;
  else
    New = SunkCopies[ExitBlock] =
        CloneInstructionInExitBlock(*I, *ExitBlock, *TPN, LI, SafetyInfo);
  return New;
}

static bool canSplitPredecessors(PHINode *PN, LoopSafetyInfo *SafetyInfo) {
  BasicBlock *BB = PN->getParent();
  if (!BB->canSplitPredecessors())
    return false;
  // It's not impossible to split EHPad blocks, but if BlockColors already exist
  // it require updating BlockColors for all offspring blocks accordingly. By
  // skipping such corner case, we can make updating BlockColors after splitting
  // predecessor fairly simple.
  if (!SafetyInfo->BlockColors.empty() && BB->getFirstNonPHI()->isEHPad())
    return false;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *BBPred = *PI;
    if (isa<IndirectBrInst>(BBPred->getTerminator()))
      return false;
  }
  return true;
}

static void splitPredecessorsOfLoopExit(PHINode *PN, DominatorTree *DT,
                                        LoopInfo *LI, const Loop *CurLoop,
                                        LoopSafetyInfo *SafetyInfo) {
#ifndef NDEBUG
  SmallVector<BasicBlock *, 32> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);
  SmallPtrSet<BasicBlock *, 32> ExitBlockSet(ExitBlocks.begin(),
                                             ExitBlocks.end());
#endif
  BasicBlock *ExitBB = PN->getParent();
  assert(ExitBlockSet.count(ExitBB) && "Expect the PHI is in an exit block.");

  // Split predecessors of the loop exit to make instructions in the loop are
  // exposed to exit blocks through trivially replacable PHIs while keeping the
  // loop in the canonical form where each predecessor of each exit block should
  // be contained within the loop. For example, this will convert the loop below
  // from
  //
  // LB1:
  //   %v1 =
  //   br %LE, %LB2
  // LB2:
  //   %v2 =
  //   br %LE, %LB1
  // LE:
  //   %p = phi [%v1, %LB1], [%v2, %LB2] <-- non-trivially replacable
  //
  // to
  //
  // LB1:
  //   %v1 =
  //   br %LE.split, %LB2
  // LB2:
  //   %v2 =
  //   br %LE.split2, %LB1
  // LE.split:
  //   %p1 = phi [%v1, %LB1]  <-- trivially replacable
  //   br %LE
  // LE.split2:
  //   %p2 = phi [%v2, %LB2]  <-- trivially replacable
  //   br %LE
  // LE:
  //   %p = phi [%p1, %LE.split], [%p2, %LE.split2]
  //
  auto &BlockColors = SafetyInfo->BlockColors;
  SmallSetVector<BasicBlock *, 8> PredBBs(pred_begin(ExitBB), pred_end(ExitBB));
  while (!PredBBs.empty()) {
    BasicBlock *PredBB = *PredBBs.begin();
    assert(CurLoop->contains(PredBB) &&
           "Expect all predecessors are in the loop");
    if (PN->getBasicBlockIndex(PredBB) >= 0) {
      BasicBlock *NewPred = SplitBlockPredecessors(
          ExitBB, PredBB, ".split.loop.exit", DT, LI, true);
      // Since we do not allow splitting EH-block with BlockColors in
      // canSplitPredecessors(), we can simply assign predecessor's color to
      // the new block.
      if (!BlockColors.empty())
        BlockColors[NewPred] = BlockColors[PredBB];
    }
    PredBBs.remove(PredBB);
  }
}

/// When an instruction is found to only be used outside of the loop, this
/// function moves it to the exit blocks and patches up SSA form as needed.
/// This method is guaranteed to remove the original instruction from its
/// position, and may either delete it or move it to outside of the loop.
///
static bool sink(Instruction &I, LoopInfo *LI, DominatorTree *DT,
                 const Loop *CurLoop, LoopSafetyInfo *SafetyInfo,
                 OptimizationRemarkEmitter *ORE, bool FreeInLoop) {
  DEBUG(dbgs() << "LICM sinking instruction: " << I << "\n");
  ORE->emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "InstSunk", &I)
           << "sinking " << ore::NV("Inst", &I);
  });
  bool Changed = false;
  if (isa<LoadInst>(I))
    ++NumMovedLoads;
  else if (isa<CallInst>(I))
    ++NumMovedCalls;
  ++NumSunk;

  // Iterate over users to be ready for actual sinking. Replace users via
  // unrechable blocks with undef and make all user PHIs trivially replcable.
  SmallPtrSet<Instruction *, 8> VisitedUsers;
  for (Value::user_iterator UI = I.user_begin(), UE = I.user_end(); UI != UE;) {
    auto *User = cast<Instruction>(*UI);
    Use &U = UI.getUse();
    ++UI;

    if (VisitedUsers.count(User) || CurLoop->contains(User))
      continue;

    if (!DT->isReachableFromEntry(User->getParent())) {
      U = UndefValue::get(I.getType());
      Changed = true;
      continue;
    }

    // The user must be a PHI node.
    PHINode *PN = cast<PHINode>(User);

    // Surprisingly, instructions can be used outside of loops without any
    // exits.  This can only happen in PHI nodes if the incoming block is
    // unreachable.
    BasicBlock *BB = PN->getIncomingBlock(U);
    if (!DT->isReachableFromEntry(BB)) {
      U = UndefValue::get(I.getType());
      Changed = true;
      continue;
    }

    VisitedUsers.insert(PN);
    if (isTriviallyReplacablePHI(*PN, I))
      continue;

    if (!canSplitPredecessors(PN, SafetyInfo))
      return Changed;

    // Split predecessors of the PHI so that we can make users trivially
    // replacable.
    splitPredecessorsOfLoopExit(PN, DT, LI, CurLoop, SafetyInfo);

    // Should rebuild the iterators, as they may be invalidated by
    // splitPredecessorsOfLoopExit().
    UI = I.user_begin();
    UE = I.user_end();
  }

  if (VisitedUsers.empty())
    return Changed;

#ifndef NDEBUG
  SmallVector<BasicBlock *, 32> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);
  SmallPtrSet<BasicBlock *, 32> ExitBlockSet(ExitBlocks.begin(),
                                             ExitBlocks.end());
#endif

  // Clones of this instruction. Don't create more than one per exit block!
  SmallDenseMap<BasicBlock *, Instruction *, 32> SunkCopies;

  // If this instruction is only used outside of the loop, then all users are
  // PHI nodes in exit blocks due to LCSSA form. Just RAUW them with clones of
  // the instruction.
  SmallSetVector<User*, 8> Users(I.user_begin(), I.user_end());
  for (auto *UI : Users) {
    auto *User = cast<Instruction>(UI);

    if (CurLoop->contains(User))
      continue;

    PHINode *PN = cast<PHINode>(User);
    assert(ExitBlockSet.count(PN->getParent()) &&
           "The LCSSA PHI is not in an exit block!");
    // The PHI must be trivially replacable.
    Instruction *New = sinkThroughTriviallyReplacablePHI(PN, &I, LI, SunkCopies,
                                                         SafetyInfo, CurLoop);
    PN->replaceAllUsesWith(New);
    PN->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

/// When an instruction is found to only use loop invariant operands that
/// is safe to hoist, this instruction is called to do the dirty work.
///
static bool hoist(Instruction &I, const DominatorTree *DT, const Loop *CurLoop,
                  const LoopSafetyInfo *SafetyInfo,
                  OptimizationRemarkEmitter *ORE) {
  auto *Preheader = CurLoop->getLoopPreheader();
  DEBUG(dbgs() << "LICM hoisting to " << Preheader->getName() << ": " << I
               << "\n");
  ORE->emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "Hoisted", &I) << "hoisting "
                                                         << ore::NV("Inst", &I);
  });

  // Metadata can be dependent on conditions we are hoisting above.
  // Conservatively strip all metadata on the instruction unless we were
  // guaranteed to execute I if we entered the loop, in which case the metadata
  // is valid in the loop preheader.
  if (I.hasMetadataOtherThanDebugLoc() &&
      // The check on hasMetadataOtherThanDebugLoc is to prevent us from burning
      // time in isGuaranteedToExecute if we don't actually have anything to
      // drop.  It is a compile time optimization, not required for correctness.
      !isGuaranteedToExecute(I, DT, CurLoop, SafetyInfo))
    I.dropUnknownNonDebugMetadata();

  // Move the new node to the Preheader, before its terminator.
  I.moveBefore(Preheader->getTerminator());

  // Do not retain debug locations when we are moving instructions to different
  // basic blocks, because we want to avoid jumpy line tables. Calls, however,
  // need to retain their debug locs because they may be inlined.
  // FIXME: How do we retain source locations without causing poor debugging
  // behavior?
  if (!isa<CallInst>(I))
    I.setDebugLoc(DebugLoc());

  if (isa<LoadInst>(I))
    ++NumMovedLoads;
  else if (isa<CallInst>(I))
    ++NumMovedCalls;
  ++NumHoisted;
  return true;
}

/// Only sink or hoist an instruction if it is not a trapping instruction,
/// or if the instruction is known not to trap when moved to the preheader.
/// or if it is a trapping instruction and is guaranteed to execute.
static bool isSafeToExecuteUnconditionally(Instruction &Inst,
                                           const DominatorTree *DT,
                                           const Loop *CurLoop,
                                           const LoopSafetyInfo *SafetyInfo,
                                           OptimizationRemarkEmitter *ORE,
                                           const Instruction *CtxI) {
  if (isSafeToSpeculativelyExecute(&Inst, CtxI, DT))
    return true;

  bool GuaranteedToExecute =
      isGuaranteedToExecute(Inst, DT, CurLoop, SafetyInfo);

  if (!GuaranteedToExecute) {
    auto *LI = dyn_cast<LoadInst>(&Inst);
    if (LI && CurLoop->isLoopInvariant(LI->getPointerOperand()))
      ORE->emit([&]() {
        return OptimizationRemarkMissed(
                   DEBUG_TYPE, "LoadWithLoopInvariantAddressCondExecuted", LI)
               << "failed to hoist load with loop-invariant address "
                  "because load is conditionally executed";
      });
  }

  return GuaranteedToExecute;
}

namespace {
class LoopPromoter : public LoadAndStorePromoter {
  Value *SomePtr; // Designated pointer to store to.
  const SmallSetVector<Value *, 8> &PointerMustAliases;
  SmallVectorImpl<BasicBlock *> &LoopExitBlocks;
  SmallVectorImpl<Instruction *> &LoopInsertPts;
  PredIteratorCache &PredCache;
  AliasSetTracker &AST;
  LoopInfo &LI;
  DebugLoc DL;
  int Alignment;
  bool UnorderedAtomic;
  AAMDNodes AATags;

  Value *maybeInsertLCSSAPHI(Value *V, BasicBlock *BB) const {
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (Loop *L = LI.getLoopFor(I->getParent()))
        if (!L->contains(BB)) {
          // We need to create an LCSSA PHI node for the incoming value and
          // store that.
          PHINode *PN = PHINode::Create(I->getType(), PredCache.size(BB),
                                        I->getName() + ".lcssa", &BB->front());
          for (BasicBlock *Pred : PredCache.get(BB))
            PN->addIncoming(I, Pred);
          return PN;
        }
    return V;
  }

public:
  LoopPromoter(Value *SP, ArrayRef<const Instruction *> Insts, SSAUpdater &S,
               const SmallSetVector<Value *, 8> &PMA,
               SmallVectorImpl<BasicBlock *> &LEB,
               SmallVectorImpl<Instruction *> &LIP, PredIteratorCache &PIC,
               AliasSetTracker &ast, LoopInfo &li, DebugLoc dl, int alignment,
               bool UnorderedAtomic, const AAMDNodes &AATags)
      : LoadAndStorePromoter(Insts, S), SomePtr(SP), PointerMustAliases(PMA),
        LoopExitBlocks(LEB), LoopInsertPts(LIP), PredCache(PIC), AST(ast),
        LI(li), DL(std::move(dl)), Alignment(alignment),
        UnorderedAtomic(UnorderedAtomic), AATags(AATags) {}

  bool isInstInList(Instruction *I,
                    const SmallVectorImpl<Instruction *> &) const override {
    Value *Ptr;
    if (LoadInst *LI = dyn_cast<LoadInst>(I))
      Ptr = LI->getOperand(0);
    else
      Ptr = cast<StoreInst>(I)->getPointerOperand();
    return PointerMustAliases.count(Ptr);
  }

  void doExtraRewritesBeforeFinalDeletion() const override {
    // Insert stores after in the loop exit blocks.  Each exit block gets a
    // store of the live-out values that feed them.  Since we've already told
    // the SSA updater about the defs in the loop and the preheader
    // definition, it is all set and we can start using it.
    for (unsigned i = 0, e = LoopExitBlocks.size(); i != e; ++i) {
      BasicBlock *ExitBlock = LoopExitBlocks[i];
      Value *LiveInValue = SSA.GetValueInMiddleOfBlock(ExitBlock);
      LiveInValue = maybeInsertLCSSAPHI(LiveInValue, ExitBlock);
      Value *Ptr = maybeInsertLCSSAPHI(SomePtr, ExitBlock);
      Instruction *InsertPos = LoopInsertPts[i];
      StoreInst *NewSI = new StoreInst(LiveInValue, Ptr, InsertPos);
      if (UnorderedAtomic)
        NewSI->setOrdering(AtomicOrdering::Unordered);
      NewSI->setAlignment(Alignment);
      NewSI->setDebugLoc(DL);
      if (AATags)
        NewSI->setAAMetadata(AATags);
    }
  }

  void replaceLoadWithValue(LoadInst *LI, Value *V) const override {
    // Update alias analysis.
    AST.copyValue(LI, V);
  }
  void instructionDeleted(Instruction *I) const override { AST.deleteValue(I); }
};


/// Return true iff we can prove that a caller of this function can not inspect
/// the contents of the provided object in a well defined program.
bool isKnownNonEscaping(Value *Object, const TargetLibraryInfo *TLI) {
  if (isa<AllocaInst>(Object))
    // Since the alloca goes out of scope, we know the caller can't retain a
    // reference to it and be well defined.  Thus, we don't need to check for
    // capture.
    return true;

  // For all other objects we need to know that the caller can't possibly
  // have gotten a reference to the object.  There are two components of
  // that:
  //   1) Object can't be escaped by this function.  This is what
  //      PointerMayBeCaptured checks.
  //   2) Object can't have been captured at definition site.  For this, we
  //      need to know the return value is noalias.  At the moment, we use a
  //      weaker condition and handle only AllocLikeFunctions (which are
  //      known to be noalias).  TODO
  return isAllocLikeFn(Object, TLI) &&
    !PointerMayBeCaptured(Object, true, true);
}

} // namespace

/// Returns an owning pointer to an alias set which incorporates aliasing info
/// from L and all subloops of L.
/// FIXME: In new pass manager, there is no helper function to handle loop
/// analysis such as cloneBasicBlockAnalysis, so the AST needs to be recomputed
/// from scratch for every loop. Hook up with the helper functions when
/// available in the new pass manager to avoid redundant computation.
AliasSetTracker *
LoopInvariantCodeMotion::collectAliasInfoForLoop(Loop *L, LoopInfo *LI,
                                                 AliasAnalysis *AA) {
  AliasSetTracker *CurAST = nullptr;
  SmallVector<Loop *, 4> RecomputeLoops;
  for (Loop *InnerL : L->getSubLoops()) {
    auto MapI = LoopToAliasSetMap.find(InnerL);
    // If the AST for this inner loop is missing it may have been merged into
    // some other loop's AST and then that loop unrolled, and so we need to
    // recompute it.
    if (MapI == LoopToAliasSetMap.end()) {
      RecomputeLoops.push_back(InnerL);
      continue;
    }
    AliasSetTracker *InnerAST = MapI->second;

    if (CurAST != nullptr) {
      // What if InnerLoop was modified by other passes ?
      CurAST->add(*InnerAST);

      // Once we've incorporated the inner loop's AST into ours, we don't need
      // the subloop's anymore.
      delete InnerAST;
    } else {
      CurAST = InnerAST;
    }
    LoopToAliasSetMap.erase(MapI);
  }
  if (CurAST == nullptr)
    CurAST = new AliasSetTracker(*AA);

  auto mergeLoop = [&](Loop *L) {
    // Loop over the body of this loop, looking for calls, invokes, and stores.
    for (BasicBlock *BB : L->blocks())
      CurAST->add(*BB); // Incorporate the specified basic block
  };

  // Add everything from the sub loops that are no longer directly available.
  for (Loop *InnerL : RecomputeLoops)
    mergeLoop(InnerL);

  // And merge in this loop.
  mergeLoop(L);

  return CurAST;
}

/// Simple analysis hook. Clone alias set info.
///
void FPLICMPass::cloneBasicBlockAnalysis(BasicBlock *From, BasicBlock *To,
                                             Loop *L) {
  AliasSetTracker *AST = LICM.getLoopToAliasSetMap().lookup(L);
  if (!AST)
    return;

  AST->copyValue(From, To);
}

/// Simple Analysis hook. Delete value V from alias set
///
void FPLICMPass::deleteAnalysisValue(Value *V, Loop *L) {
  AliasSetTracker *AST = LICM.getLoopToAliasSetMap().lookup(L);
  if (!AST)
    return;

  AST->deleteValue(V);
}

/// Simple Analysis hook. Delete value L from alias set map.
///
void FPLICMPass::deleteAnalysisLoop(Loop *L) {
  AliasSetTracker *AST = LICM.getLoopToAliasSetMap().lookup(L);
  if (!AST)
    return;

  delete AST;
  LICM.getLoopToAliasSetMap().erase(L);
}

/// Return true if the body of this loop may store into the memory
/// location pointed to by V.
///
static bool pointerInvalidatedByLoop(Value *V, uint64_t Size,
                                     const AAMDNodes &AAInfo,
                                     AliasSetTracker *CurAST) {
  // Check to see if any of the basic blocks in CurLoop invalidate *V.
  return CurAST->getAliasSetForPointer(V, Size, AAInfo).isMod();
}

/// Little predicate that returns true if the specified basic block is in
/// a subloop of the current one, not the current one itself.
///
static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI) {
  assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
  return LI->getLoopFor(BB) != CurLoop;
}
