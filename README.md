# Trust Region and Riccati Recursion for NMPC

## Objective
The main objective of this project was to develop a solver able to handle Non-linear Model Predictive Control
(NMPC) problems using trust-region method. The method uses Sequential Quadratic Programming (SQP)
but solves the Quadatric Programming (QP) sub-problem using an improved version of Riccati recursion
that incorporate a trust-region criterion. After each forward and backward propagation, the solver will verify
that the solution is inside the trust radius. If that is not the case, a modification of the eigenvalues of the Hessian will try
to enforce the solution to remain within the trust radius.

## Implementation
The whole code has been implemented in Matlab, using two classes. The first one (riccati TR) handles the
Riccati recursion with trust region constrained. It is accessible by the second one (NOCP), which handles
the matrices construction, as well as the main solver’s loop which update the trust radius and decide when
to stop. [TestAlgorithm_riccati_TR.m](TestAlgorithm_riccati_TR.m) shows how these classes can be used to solve MPC problem for switching-time system. This
latter also computes automatically the different gradients and Hessian for each sub-systems (in the case of
switched systems).

For more details, please refer to the [report](NMPC_Report_MonnetStephen.pdf) and [presentation](NMPC_Presentation_MonnetStephen.pdf).


