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
    else {
        return nullptr;
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
        auto initializer = ConstantZero::get(tp, builder->get_module());
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
    scope.enter();

    for (auto &decl : node.local_declarations) {
        decl->accept(*this);
    }

    for (auto &stmt : node.statement_list) {
        stmt->accept(*this);
        if (builder->get_insert_block()->is_terminated())
            break;
    }

    scope.exit();
    return nullptr;
}

Value* CminusfBuilder::visit(ASTExpressionStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.
    
    if(node.expression != nullptr){
        node.expression -> accept(*this);
    }
    
    return nullptr;
}

Value* CminusfBuilder::visit(ASTSelectionStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.

    if(node.expression != nullptr){
        node.expression -> accept(*this);
    }
    Value *cond_val = context.Num;
    if(context.NumType == TYPE_INT){
        cond_val = builder-> create_icmp_ne(cond_val, CONST_INT(0));
    }
    else if(context.NumType == TYPE_FLOAT){
        cond_val = builder-> create_fcmp_ne(cond_val, CONST_FP(0.0));
    }
    else{
        cond_val = nullptr;
    }

    auto *func = context.func;
    auto *true_bb = BasicBlock::create(module.get(), "if.true" + std::to_string(context.count++), func);
    auto *false_bb = BasicBlock::create(module.get(), "if.false" + std::to_string(context.count++), func);
    auto *exit_bb = BasicBlock::create(module.get(), "if.exit" + std::to_string(context.count++), func);

    if(node.else_statement){
        builder-> create_cond_br(cond_val, true_bb, false_bb);
    }
    else {
        builder-> create_cond_br(cond_val, true_bb, exit_bb);
    }

    builder-> set_insert_point(true_bb);
    node.if_statement -> accept(*this);
    if(!builder->get_insert_block()->is_terminated()){
        builder-> create_br(exit_bb);
    }

    if(node.else_statement){
        builder-> set_insert_point(false_bb);
        node.else_statement -> accept(*this);
        if(!builder->get_insert_block()->is_terminated()){
            builder-> create_br(exit_bb);
        }
    }
    else {
        false_bb->erase_from_parent();
    }

    builder-> set_insert_point(exit_bb);

    return nullptr;
}

Value* CminusfBuilder::visit(ASTIterationStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.
    auto *func = context.func;
    auto *cond_bb = BasicBlock::create(module.get(), "while.cond" + std::to_string(context.count++), func);
    auto *loop_bb = BasicBlock::create(module.get(), "while.loop" + std::to_string(context.count++), func);
    auto *exit_bb = BasicBlock::create(module.get(), "while.exit" + std::to_string(context.count++), func);

    builder-> create_br(cond_bb);
    builder-> set_insert_point(cond_bb);

    if(node.expression != nullptr){
        node.expression -> accept(*this);
    }
    Value *cond_val = context.Num;
    if(context.NumType == TYPE_INT){
        cond_val = builder-> create_icmp_ne(cond_val, CONST_INT(0));
    }
    else if(context.NumType == TYPE_FLOAT){
        cond_val = builder-> create_fcmp_ne(cond_val, CONST_FP(0.0));
    }
    else{
        cond_val = nullptr;
    }

    builder-> create_cond_br(cond_val, loop_bb, exit_bb);
    builder-> set_insert_point(loop_bb);
    node.statement -> accept(*this);
    if(!builder->get_insert_block()->is_terminated()){
        builder-> create_br(cond_bb);
    }

    builder-> set_insert_point(exit_bb);

    return nullptr;
}

Value* CminusfBuilder::visit(ASTReturnStmt &node) {
    if (node.expression == nullptr) {
        builder->create_void_ret();
        return nullptr;
    } else {
        // TODO: The given code is incomplete.
        // You need to solve other return cases (e.g. return an integer).
        node.expression->accept(*this);
        Value *return_val = context.Num;

        Type *expected_ret_type = context.func->get_return_type();

        if(expected_ret_type->is_float_type() && context.NumType == TYPE_INT){
            return_val = builder->create_sitofp(return_val, FLOAT_T);
            context.NumType = TYPE_FLOAT;
        }
        else if(expected_ret_type->is_int32_type() && context.NumType == TYPE_FLOAT){
            return_val = builder->create_fptosi(return_val, INT32_T);
            context.NumType = TYPE_INT;
        }

        builder->create_ret(return_val); 
    }
    return nullptr;  
}

Value* CminusfBuilder::visit(ASTVar &node) {
    // TODO: This function is empty now.
    // Add some code here.
    auto var_ptr = scope.find(node.id);
    assert(var_ptr != nullptr && "Variable not found in the scope");
    if(node.expression == nullptr){
        auto pointee_type = var_ptr->get_type()->get_pointer_element_type();

        if(pointee_type->is_integer_type()|| pointee_type->is_float_type()){
            context.varAddr = var_ptr;
            context.NumType = (pointee_type->is_integer_type()) ? TYPE_INT : TYPE_FLOAT;
            context.Num = builder->create_load(var_ptr);
        }

        else if(pointee_type->is_array_type()){
            context.varAddr = builder->create_gep(var_ptr, std::vector<Value *>{CONST_INT(0), CONST_INT(0)});
        }

        else if(pointee_type->is_pointer_type()){
            context.varAddr = builder->create_load(var_ptr);
        }
    }

    else{
        node.expression->accept(*this);
        Value *idx = context.Num;
        if(context.NumType == TYPE_FLOAT){
            idx = builder->create_fptosi(idx, INT32_T);
            context.NumType = TYPE_INT;
        }

        auto *func = context.func;
        auto *check_bb = BasicBlock::create(module.get(), "arrayidx.check" + std::to_string(context.count++), func);
        auto *error_bb = BasicBlock::create(module.get(), "arrayidx.error" + std::to_string(context.count++), func);

        Value *is_non_negative = builder->create_icmp_sge(idx, CONST_INT(0)); // >=0
        builder->create_cond_br(is_non_negative, check_bb, error_bb);

        builder->set_insert_point(error_bb);
        auto *neg_except_fun = scope.find("neg_idx_except");
        builder->create_call(neg_except_fun, {});
        builder->create_br(check_bb);
        
        builder->set_insert_point(check_bb);
        Value *array_base_ptr;
        auto pointee_type = var_ptr->get_type()->get_pointer_element_type();
        if(pointee_type->is_array_type()){
            array_base_ptr = var_ptr;
            context.varAddr = builder->create_gep(array_base_ptr, std::vector<Value *>{CONST_INT(0), idx});
        }

        else if(pointee_type->is_pointer_type()){
            array_base_ptr = builder->create_load(var_ptr);
            context.varAddr = builder->create_gep(array_base_ptr, {idx});
        }

        context.Num = builder->create_load(context.varAddr);
        auto element_type = context.varAddr->get_type()->get_pointer_element_type();
        context.NumType = (element_type->is_integer_type()) ? TYPE_INT : TYPE_FLOAT;
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTAssignExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    node.var->accept(*this);
    Value *l_addr = context.varAddr;

    node.expression->accept(*this);
    Value *r_val = context.Num;

    Type *l_type = l_addr->get_type()->get_pointer_element_type();
    
    if(l_type->is_integer_type() && context.NumType == TYPE_FLOAT){
        r_val = builder->create_fptosi(r_val, INT32_T);
        context.NumType = TYPE_INT;
    }
    else if(l_type->is_float_type() && context.NumType == TYPE_INT){
        r_val = builder->create_sitofp(r_val, FLOAT_T);
        context.NumType = TYPE_FLOAT;
    }
    
    builder->create_store(r_val, l_addr);
    context.Num = r_val;

    return nullptr;
}

Value* CminusfBuilder::visit(ASTSimpleExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    if(node.additive_expression_r == nullptr){
        node.additive_expression_l->accept(*this);
        return nullptr;
    }

    node.additive_expression_l->accept(*this);
    Value *l_val = context.Num;
    CminusType l_type = context.NumType;
    node.additive_expression_r->accept(*this);
    Value *r_val = context.Num;
    CminusType r_type = context.NumType;

    Value *res_val = nullptr;

    if(l_type == TYPE_FLOAT || r_type == TYPE_FLOAT){
        if(l_type == TYPE_INT){
            l_val = builder->create_sitofp(l_val, FLOAT_T);
        }
        if(r_type == TYPE_INT){
            r_val = builder->create_sitofp(r_val, FLOAT_T);
        }

        switch (node.op)
        {
        case OP_LT:
            res_val = builder->create_fcmp_lt(l_val, r_val);
            break;
        case OP_LE:
            res_val = builder->create_fcmp_le(l_val, r_val);
            break;
        case OP_GT:
            res_val = builder->create_fcmp_gt(l_val, r_val);
            break;
        case OP_GE:
            res_val = builder->create_fcmp_ge(l_val, r_val);
            break;
        case OP_EQ:
            res_val = builder->create_fcmp_eq(l_val, r_val);
            break;
        case OP_NEQ:
            res_val = builder->create_fcmp_ne(l_val, r_val);
            break;
        }
    }
    else{
        switch (node.op)
        {
        case OP_LT:
            res_val = builder->create_icmp_lt(l_val, r_val);
            break;
        case OP_LE:
            res_val = builder->create_icmp_le(l_val, r_val);
            break;
        case OP_GT:
            res_val = builder->create_icmp_gt(l_val, r_val);
            break;
        case OP_GE:
            res_val = builder->create_icmp_ge(l_val, r_val);
            break;
        case OP_EQ:
            res_val = builder->create_icmp_eq(l_val, r_val);
            break;
        case OP_NEQ:
            res_val = builder->create_icmp_ne(l_val, r_val);
            break;
        }    
    }

    context.Num = builder->create_zext(res_val, INT32_T);
    context.NumType = TYPE_INT;

    return nullptr;
}

Value* CminusfBuilder::visit(ASTAdditiveExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    if(node.additive_expression == nullptr){
        node.term->accept(*this);
        return nullptr;
    }

    node.additive_expression->accept(*this);
    Value *l_val = context.Num;
    CminusType l_type = context.NumType;

    node.term->accept(*this);
    Value *r_val = context.Num;
    CminusType r_type = context.NumType;

    if(l_type == TYPE_FLOAT || r_type == TYPE_FLOAT){
        if(l_type == TYPE_INT){
            l_val = builder->create_sitofp(l_val, FLOAT_T);
        }
        if(r_type == TYPE_INT){
            r_val = builder->create_sitofp(r_val, FLOAT_T);
        }

        switch (node.op)
        {
        case OP_PLUS:
            context.Num = builder->create_fadd(l_val, r_val);
            break;
        case OP_MINUS:
            context.Num = builder->create_fsub(l_val, r_val);
            break;
        }
        context.NumType = TYPE_FLOAT;
    }
    else{
        switch (node.op)
        {
        case OP_PLUS:
            context.Num = builder->create_iadd(l_val, r_val);
            break;
        case OP_MINUS:
            context.Num = builder->create_isub(l_val, r_val);
            break;
        }
        context.NumType = TYPE_INT;
    }

    return nullptr;
}

Value* CminusfBuilder::visit(ASTTerm &node) {
    // TODO: This function is empty now.
    // Add some code here.
    if(node.term == nullptr){
        node.factor->accept(*this);
        return nullptr;
    }

    node.term->accept(*this);
    Value *l_val = context.Num;
    CminusType l_type = context.NumType;

    node.factor->accept(*this);
    Value *r_val = context.Num;
    CminusType r_type = context.NumType;

    if(l_type == TYPE_FLOAT || r_type == TYPE_FLOAT){
        if(l_type == TYPE_INT){
            l_val = builder->create_sitofp(l_val, FLOAT_T);
        }
        if(r_type == TYPE_INT){
            r_val = builder->create_sitofp(r_val, FLOAT_T);
        }

        switch (node.op)
        {
        case OP_MUL:
            context.Num = builder->create_fmul(l_val, r_val);
            break;
        case OP_DIV:
            context.Num = builder->create_fdiv(l_val, r_val);
            break;
        }
        context.NumType = TYPE_FLOAT;
    }

    else{
        switch (node.op)
        {
        case OP_MUL:
            context.Num = builder->create_imul(l_val, r_val);
            break;
        case OP_DIV:
            context.Num = builder->create_isdiv(l_val, r_val);
            break;
        }
        context.NumType = TYPE_INT;
    }

    return nullptr;
}

Value* CminusfBuilder::visit(ASTCall &node) {
    // TODO: This function is empty now.
    // Add some code here.
    auto callee_fun = scope.find(node.id);
    assert(callee_fun != nullptr && "Function not found in the scope");
    auto func = static_cast<Function *>(callee_fun);
    std::vector<Value *> args;

    for(int i = 0; i < node.args.size(); ++i){
        node.args[i]->accept(*this);
        Value *arg_val = context.Num;
        CminusType arg_type = context.NumType;

        Type *param_type = func->get_function_type()->get_param_type(i);
        if(param_type->is_integer_type() && arg_type == TYPE_FLOAT){
            arg_val = builder->create_fptosi(arg_val, INT32_T);
            context.NumType = TYPE_INT;
        }
        else if(param_type->is_float_type() && arg_type == TYPE_INT){
            arg_val = builder->create_sitofp(arg_val, FLOAT_T);
            context.NumType = TYPE_FLOAT;
        }

        args.push_back(arg_val);
    }

    context.Num = builder->create_call(func, args);

    auto ret_type = func->get_return_type();
    if(ret_type->is_integer_type()){
        context.NumType = TYPE_INT;
    }
    else if(ret_type->is_float_type()){
        context.NumType = TYPE_FLOAT;
    }
    else if(ret_type->is_void_type()){
        context.NumType = TYPE_VOID;
    }

    return nullptr;
}
