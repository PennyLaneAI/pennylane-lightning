
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif
 
void
__cyg_profile_func_enter (void *func,  void *caller)
{
    std::cerr << "Function entered \n";
}
 
void
__cyg_profile_func_exit (void *func, void *caller)
{
    std::cerr << "Function exited \n";
}

#ifdef __cplusplus
}
#endif