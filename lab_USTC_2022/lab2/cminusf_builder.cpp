#include "cminusf_builder.hpp"

#define CONST_FP(num) ConstantFP::get((float)num, module.get())
#define CONST_INT(num) ConstantInt::get(num, module.get())

// types
Type *VOID_T;
Type *INT1_T;
Type *INT32_T;
Type *INT32PTR_T;
Type *FLOAT_T;
Type *FLOATPTR_T;

/*
 * use CMinusfBuilder::Scope to construct scopes
 * scope.enter: enter a new scope
 * scope.exit: exit current scope
 * scope.push: add a new binding to current scope
 * scope.find: find and return the value bound to the name
 */

Value* CminusfBuilder::visit(ASTProgram &node) {
    VOID_T = module->get_void_type();
    INT1_T = module->get_int1_type();
    INT32_T = module->get_int32_type();
    INT32PTR_T = module->get_int32_ptr_type();
    FLOAT_T = module->get_float_type();
    FLOATPTR_T = module->get_float_ptr_type();

    Value *ret_val = nullptr;
    for (auto &decl : node.declarations) {
        ret_val = decl->accept(*this);
    }
    return ret_val;
}

Value* CminusfBuilder::visit(ASTNum &node) {
    // TODO: This function is empty now.
    // Add some code here.
    if (node.type == TYPE_INT) {
        context.NumType = TYPE_INT;
        context.Num = CONST_INT(node.i_val);
        context.INTEGER = node.i_val;
    }
    else if (node.type == TYPE_FLOAT) {
        context.NumType = TYPE_FLOAT;
        context.Num = CONST_FP(node.f_val);
    }
    else {
        context.NumType = TYPE_VOID;
        context.Num = nullptr;
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTVarDeclaration &node) {
    // TODO: This function is empty now.
    // Add some code here.
    Type *tp = nullptr;
    if (node.type == TYPE_INT)
    {
        tp = INT32_T;
    }
    else if (node.type == TYPE_FLOAT)
    {
        tp = FLOAT_T;
    }
    if (not scope.in_global())
    {
        if (node.num == nullptr)
        {
            auto Alloca = (tp != nullptr) ? builder->create_alloca(tp) : nullptr;
            scope.push(node.id, Alloca);
        }
        else
        {
            node.num->accept(*this);
            if (context.INTEGER <= 0)
                builder->create_call(scope.find("neg_idx_except"), std::vector<Value *>{});
            auto arrytype = ArrayType::get(tp, context.INTEGER);
            auto arryAllca = builder->create_alloca(arrytype);
            scope.push(node.id, arryAllca);
        }
    }
    else
    {
        auto initializer = ConstantZero::get(INT32_T, builder->get_module());
        if (node.num == nullptr)
        {

            auto Alloca = (tp != nullptr) ? GlobalVariable::create(node.id, builder->get_module(), tp, false, initializer) : nullptr;
            scope.push(node.id, Alloca);
        }
        else
        {
            node.num->accept(*this);
            if (context.INTEGER <= 0)
                builder->create_call(scope.find("neg_idx_except"), std::vector<Value *>{});
            auto arrytype = ArrayType::get(tp, context.INTEGER);
            auto arryAllca = GlobalVariable::create(node.id, builder->get_module(), arrytype, false, initializer);
            scope.push(node.id, arryAllca);
        }
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTFunDeclaration &node) {
    FunctionType *fun_type;
    Type *ret_type;
    std::vector<Type *> param_types;
    std::vector<std::string> param_id;
    if (node.type == TYPE_INT)
        ret_type = INT32_T;
    else if (node.type == TYPE_FLOAT)
        ret_type = FLOAT_T;
    else
        ret_type = VOID_T;

    for (auto &param : node.params) {
        // TODO: Please accomplish param_types.
        param->accept(*this);
        param_types.push_back(context.ParaType);
        param_id.push_back(context.param_id);
    }

    fun_type = FunctionType::get(ret_type, param_types);
    auto func = Function::create(fun_type, node.id, module.get());
    scope.push(node.id, func);
    context.func = func;
    auto funBB = BasicBlock::create(module.get(), "entry", func);
    builder->set_insert_point(funBB);
    scope.enter();
    std::vector<Value *> args;
    for (auto &arg : func->get_args()) {
        args.push_back(&arg);
    }
    for (int i = 0; i < node.params.size(); ++i) {
        // TODO: You need to deal with params and store them in the scope.
        auto argAlloca = builder->create_alloca(args[i]->get_type());
        builder->create_store(args[i], argAlloca);
        scope.push(param_id[i], argAlloca);
    }
    node.compound_stmt->accept(*this);
    if (not builder->get_insert_block()->is_terminated())
    {
        if (context.func->get_return_type()->is_void_type())
            builder->create_void_ret();
        else if (context.func->get_return_type()->is_float_type())
            builder->create_ret(CONST_FP(0.));
        else
            builder->create_ret(CONST_INT(0));
    }
    scope.exit();
    return nullptr;
}

Value* CminusfBuilder::visit(ASTParam &node) {
    // TODO: This function is empty now.
    // Add some code here.
     context.param_id=node.id;
    if (node.isarray)
    {
        if (node.type == TYPE_INT)
        {
            context.ParaType = PointerType::get(INT32_T);
        }
        else if (node.type == TYPE_FLOAT)
        {
            context.ParaType = PointerType::get(FLOAT_T);
        }
        else
        {
            context.ParaType = PointerType::get(VOID_T);
        }
    }
    else
    {
        if (node.type == TYPE_INT)
        {
            context.ParaType = INT32_T;
        }
        else if (node.type == TYPE_FLOAT)
        {
            context.ParaType = FLOAT_T;
        }
        else
        {
            context.ParaType = VOID_T;
        }
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTCompoundStmt &node) {
    // TODO: This function is not complete.
    // You may need to add some code here
    // to deal with complex statements.

    for (auto &decl : node.local_declarations) {
        decl->accept(*this);
    }

    for (auto &stmt : node.statement_list) {
        stmt->accept(*this);
        if (builder->get_insert_block()->is_terminated())
            break;
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTExpressionStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTSelectionStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTIterationStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTReturnStmt &node) {
    if (node.expression == nullptr) {
        builder->create_void_ret();
        return nullptr;
    } else {
        // TODO: The given code is incomplete.
        // You need to solve other return cases (e.g. return an integer).
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTVar &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTAssignExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTSimpleExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTAdditiveExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTTerm &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTCall &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}
