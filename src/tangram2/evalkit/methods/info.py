# from telegraph.methods.workflows import WorkFlowClass

# from . import constants as C


# def _list_options(object_name: str):
#     # base function to list the different options
#     # for our different components
#     options = eval("C.{}.get_options()".format(object_name.upper()))
#     print_str = "{}:\n  - ".format(object_name) + "\n  - ".join(options)
#     print(print_str)


# def list_methods():
#     """lists methods"""
#     _list_options("methods")


# def list_metrics():
#     """lists metrics"""
#     _list_options("metrics")


# def list_pp():
#     """lists preprocessing methods"""
#     _list_options("preprocess")


# def list_workflows():
#     """lists workflows"""
#     _list_options("workflows")


# def list_chained_methods(
#     wf: WorkFlowClass,
# ) -> None:
#     """list methods of a workflow"""
#     print("Elements of {}:".format(wf.__name__))
#     for k, method_name in enumerate(wf.flow.methods):
#         print("  - {} : {}".format(k, method_name))
