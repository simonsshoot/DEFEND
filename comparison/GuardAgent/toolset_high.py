import sys
import traceback
from io import StringIO
from io import StringIO

'''代码执行器函数，专门用于执行 GuardAgent 为 EHR（电子健康记录）Agent 生成的守护代码，并返回访问控制结果。
cell: AI Agent 生成的 Python 代码字符串（守护代码）
执行代码：exec
CodeHeader: 从 prompts_guard 导入的代码头部（包含必要的工具函数定义）
执行组合后的完整代码
结果存储在 global_var 字典中
'''
def run_code_ehragent(cell):
    """
    Returns the path to the python interpreter.
    """
    from prompts_guard import CodeHeader
    try:
        global_var = {"access_denied": None,
                      "inaccessible_database": None,
                      "guardrailed_answer": None}
        exec(CodeHeader + cell, global_var)
        cell = "\n".join([line for line in cell.split("\n") if line.strip() and not line.strip().startswith("#")])
        if not ('guardrailed_answer' in cell and 'access_denied' in cell and 'inaccessible_database' in cell):
            return "Missing variables."
        return "GuardAgent results:\nlabel: {}\ninaccessible_db: {}\nguardrailed_answer: {}\n(End of results)".format(int(global_var['access_denied']), global_var['inaccessible_database'], global_var['guardrailed_answer'])
    except Exception as e:
        error_info = traceback.format_exc()
        code = CodeHeader + cell
        if "SyntaxError" in str(repr(e)):
            error_line = str(repr(e))

            error_type = error_line.split('(')[0]
            # then parse out the error message
            error_message = error_line.split(',')[0].split('(')[1]
            # then parse out the error line
            error_line = error_line.split('"')[1]
        elif "KeyError" in str(repr(e)):
            code = code.split('\n')
            key = str(repr(e)).split("'")[1]
            error_type = str(repr(e)).split('(')[0]
            for i in range(len(code)):
                if key in code[i]:
                    error_line = code[i]
            error_message = str(repr(e))
        elif "TypeError" in str(repr(e)):
            error_type = str(repr(e)).split('(')[0]
            error_message = str(e)
            function_mapping_dict = {"check_access": "CheckAccess",
                                     "check_action": "CheckAction"}
            error_key = ""
            for key in function_mapping_dict.keys():
                if key in error_message:
                    error_message = error_message.replace(key, function_mapping_dict[key])
                    error_key = function_mapping_dict[key]
            code = code.split('\n')
            error_line = ""
            for i in range(len(code)):
                if error_key in code[i]:
                    error_line = code[i]
        else:
            error_type = ""
            error_message = str(repr(e)).split("('")[-1].split("')")[0]
            error_line = ""
        # use one sentence to introduce the previous parsed error information
        if error_type != "" and error_line != "":
            error_info = f'{error_type}: {error_message}. The error messages occur in the code line "{error_line}".'
        else:
            error_info = f'Error: {error_message}.'
        error_info += '\nPlease make modifications accordingly and make sure the rest code works well with the modification.'

        return error_info


def run_code_seeact(cell):
    """
    Returns the path to the python interpreter.
    """
    from prompts_guard import CodeHeader
    try:
        global_var = {"action_denied": None,
                      "inaccessible_actions": None,
                      "guardrailed_answer": None}
        # get the print info of the code
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        exec(CodeHeader + cell, global_var)
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        rules = [
            ["User must be a member to shop", "member"],
            ["Unvaccinated user cannot book a flight", "flight"],
            ["User without a driver's license cannot buy or rent a car","license","car"],
            ["User aged under 18 cannot book a hotel","hotel"],
            ["User must be in certain countries to search movies/musics/video","movies","music","video"],
            ["User under 15 cannot apply for jobs","jobs"]
        ]
        
        '''
        cell = "\n".join([line for line in cell.split(
            "\n") if line.strip() and not line.strip().startswith("#")])
        if not ('guardrailed_answer' in cell and 'action_denied' in cell and 'inaccessible_actions' in cell):
            return "Missing variables."
        '''
        if not ('ACTION DENIED' in output or "ACTION GRANTED" in output):
            return "Missing variables."
        if "ACTION DENIED" in output:
            action_denied = True
            guardrailed_answer = ""
            inaccessible_actions = ""
            for r in rules:
                for i in r:
                    if i in output:
                        guardrailed_answer = r[0]
                        inaccessible_actions = ""
        else:
            action_denied = False
            guardrailed_answer = output.split("guardrailed_answer:")[1]
            inaccessible_actions = ""
        return "GuardAgent results:\naction_denied: {}\ninaccessible_actions: {}\nguardrailed_answer: {}\n(End of results)".format(int(action_denied), inaccessible_actions, guardrailed_answer)
    except Exception as e:
        error_info = traceback.format_exc()
        code = CodeHeader + cell
        if "SyntaxError" in str(repr(e)):
            error_line = str(repr(e))

            error_type = error_line.split('(')[0]
            # then parse out the error message
            error_message = error_line.split(',')[0].split('(')[1]
            # then parse out the error line
            error_line = error_line.split('"')[1]
        elif "KeyError" in str(repr(e)):
            code = code.split('\n')
            key = str(repr(e)).split("'")[1]
            error_type = str(repr(e)).split('(')[0]
            for i in range(len(code)):
                if key in code[i]:
                    error_line = code[i]
            error_message = str(repr(e))
        elif "TypeError" in str(repr(e)):
            error_type = str(repr(e)).split('(')[0]
            error_message = str(e)
            function_mapping_dict = {"check_access": "CheckAccess",
                                     "check_action": "CheckAction",
                                     "check_rule": "CheckRule"}
            error_key = ""
            for key in function_mapping_dict.keys():
                if key in error_message:
                    error_message = error_message.replace(
                        key, function_mapping_dict[key])
                    error_key = function_mapping_dict[key]
            code = code.split('\n')
            error_line = ""
            for i in range(len(code)):
                if error_key in code[i]:
                    error_line = code[i]
        else:
            error_type = ""
            error_message = str(repr(e)).split("('")[-1].split("')")[0]
            error_line = ""
        # use one sentence to introduce the previous parsed error information
        if error_type != "" and error_line != "":
            error_info = f'{error_type}: {error_message}. The error messages occur in the code line "{error_line}".'
        else:
            error_info = f'Error: {error_message}.'
        error_info += '\nPlease make modifications accordingly and make sure the rest code works well with the modification.'

        return error_info


'''
关于执行，例子：
# CodeHeader 的内容（来自 prompts_guard.py）
CodeHeader = """import tools
CheckAccess = tools.check_access
CheckRule = tools.check_rule
"""

# cell 是 AI 生成的代码，例如：
cell = """
# 检查用户访问权限
allowed_dbs = ['public_data']
needed_dbs = ['patient_records', 'diagnosis']
access_denied = CheckAccess(allowed_dbs, needed_dbs)

if access_denied:
    inaccessible_database = ['patient_records', 'diagnosis']
    guardrailed_answer = "Access denied to sensitive medical data"
else:
    inaccessible_database = []
    guardrailed_answer = original_answer
"""

# 拼接后的完整代码：
complete_code = CodeHeader + cell
# 结果是：
"""
import tools
CheckAccess = tools.check_access
CheckRule = tools.check_rule

# 检查用户访问权限
allowed_dbs = ['public_data']
needed_dbs = ['patient_records', 'diagnosis']
access_denied = CheckAccess(allowed_dbs, needed_dbs)

if access_denied:
    inaccessible_database = ['patient_records', 'diagnosis']
    guardrailed_answer = "Access denied to sensitive medical data"
else:
    inaccessible_database = []
    guardrailed_answer = original_answer
"""
执行在当前python的进程空间中，结果存储在 global_var 字典中
原来还可以这样，不用大模型模拟沙盒，不过感觉这样做有些工具可能实现不了


这段代码充当代码执行引擎，对应论文第4.3节“Guardrail Code Generation and Execution”。其核心作用是动态执行GuardAgent生成的Python护栏代码，并返回安全审查结果（如是否拒绝动作、详细原因等）

可靠性：通过执行代码而非纯文本推理，确保安全决策的确定性。准确性：直接运行生成的代码，避免了语言模型在理解复杂逻辑时可能出现的误差。
'''