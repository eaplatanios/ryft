use std::ops::{Add, Mul, Neg};

use indoc::indoc;

use crate::{
    parameters::{Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::*,
};

pub(crate) fn assert_reference_scalar_sine_jit_rendering() {
    let mut context = ();
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) =
        jit(&mut context, |_, x: JitTracer<f64>| x.sin(), 2.0f64).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = sin %0
            in (%1)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_reference_graph_rendering() {
    let mut builder = GraphBuilder::<std::sync::Arc<dyn Op<f64>>, f64>::new();
    let x = builder.add_input(&1.0f64);
    let three = builder.add_constant(3.0f64);
    let sum = builder.add_equation(std::sync::Arc::new(AddOp), vec![x, three]).unwrap()[0];
    let graph = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);

    assert_eq!(
        graph.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = const
                %2:f64[] = add %0 %1
            in (%2)
        "}
        .trim_end(),
    );
}

fn bilinear_sin<Context, T>(_: &mut Context, inputs: (T, T)) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    inputs.0.clone() * inputs.1 + inputs.0.sin()
}

fn quadratic_plus_sin<Context, T>(_: &mut Context, x: T) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() + x.sin()
}

fn quartic_plus_sin<Context, T>(_: &mut Context, x: T) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
}

fn first_derivative<Context, V>(context: &mut Context, x: V) -> V
where
    V: TraceValue
        + OneLike
        + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<Linearized<V>>,
{
    grad(context, quartic_plus_sin, x).expect("first derivative should be computable")
}

fn second_derivative<Context, V>(context: &mut Context, x: V) -> V
where
    V: TraceValue
        + OneLike
        + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<Linearized<V>>,
{
    grad(context, first_derivative, x).expect("second derivative should be computable")
}

fn third_derivative<Context, V>(context: &mut Context, x: V) -> V
where
    V: TraceValue
        + OneLike
        + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<Linearized<V>>,
{
    grad(context, second_derivative, x).expect("third derivative should be computable")
}

fn fourth_derivative<Context, V>(context: &mut Context, x: V) -> V
where
    V: TraceValue
        + OneLike
        + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<Linearized<V>>,
{
    grad(context, third_derivative, x).expect("fourth derivative should be computable")
}

fn hessian_style_second_derivative<Context, V>(context: &mut Context, x: V) -> V
where
    V: TraceValue
        + OneLike
        + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<Linearized<V>>,
{
    let (_, second_derivative) =
        jvp(context, first_derivative, x.clone(), x.one_like()).expect("forward-over-reverse Hessian should succeed");
    second_derivative
}

pub(crate) fn assert_bilinear_pushforward_rendering() {
    let mut context = ();
    let (_, pushforward): (f64, LinearProgram<f64, (f64, f64), f64>) =
        linearize(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();

    assert_eq!(
        pushforward.to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[] .
            let %2:f64[] = scale %0
                %3:f64[] = scale %1
                %4:f64[] = add %2 %3
                %5:f64[] = scale %0
                %6:f64[] = add %4 %5
            in (%6)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_bilinear_pullback_rendering() {
    let mut context = ();
    let (_, pullback): (f64, LinearProgram<f64, f64, (f64, f64)>) =
        vjp(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();

    assert_eq!(
        pullback.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = scale %0
                %2:f64[] = scale %0
                %3:f64[] = scale %0
                %4:f64[] = add %1 %3
                %5:f64[] = const
            in (%4, %2)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_bilinear_jit_rendering() {
    let mut context = ();
    let (_, compiled): (f64, CompiledFunction<f64, (f64, f64), f64>) =
        jit(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[] .
            let %2:f64[] = mul %0 %1
                %3:f64[] = sin %0
                %4:f64[] = add %2 %3
            in (%4)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_quadratic_pushforward_rendering() {
    let mut context = ();
    let (_, pushforward): (f64, LinearProgram<f64, f64, f64>) =
        linearize(&mut context, quadratic_plus_sin, 2.0f64).unwrap();

    assert_eq!(
        pushforward.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = scale %0
                %2:f64[] = scale %0
                %3:f64[] = add %1 %2
                %4:f64[] = scale %0
                %5:f64[] = add %3 %4
            in (%5)
        "}
        .trim_end(),
    );
}
pub(crate) fn assert_hessian_style_second_derivative_jit_rendering() {
    let mut context = ();
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) =
        jit(&mut context, hessian_style_second_derivative, 2.0f64).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = const
                %2:f64[] = const
                %3:f64[] = const
                %4:f64[] = const
                %5:f64[] = mul %0 %0
                %6:f64[] = mul %5 %0
                %7:f64[] = mul %6 %0
                %8:f64[] = sin %0
                %9:f64[] = cos %0
                %10:f64[] = cos %0
                %11:f64[] = sin %0
                %12:f64[] = add %7 %8
                %13:f64[] = const
                %14:f64[] = const
                %15:f64[] = mul %10 %13
                %16:f64[] = mul %6 %13
                %17:f64[] = add %15 %16
                %18:f64[] = mul %0 %13
                %19:f64[] = mul %5 %18
                %20:f64[] = add %17 %19
                %21:f64[] = mul %0 %18
                %22:f64[] = mul %0 %21
                %23:f64[] = add %20 %22
                %24:f64[] = mul %0 %21
                %25:f64[] = add %23 %24
                %26:f64[] = mul %0 %1
                %27:f64[] = mul %0 %1
                %28:f64[] = add %26 %27
                %29:f64[] = mul %0 %28
                %30:f64[] = mul %5 %1
                %31:f64[] = add %29 %30
                %32:f64[] = mul %0 %31
                %33:f64[] = mul %6 %1
                %34:f64[] = add %32 %33
                %35:f64[] = mul %9 %1
                %36:f64[] = mul %11 %1
                %37:f64[] = neg %36
                %38:f64[] = add %34 %35
                %39:f64[] = mul %13 %37
                %40:f64[] = mul %10 %14
                %41:f64[] = add %39 %40
                %42:f64[] = mul %13 %31
                %43:f64[] = mul %6 %14
                %44:f64[] = add %42 %43
                %45:f64[] = add %41 %44
                %46:f64[] = mul %13 %1
                %47:f64[] = mul %0 %14
                %48:f64[] = add %46 %47
                %49:f64[] = mul %18 %28
                %50:f64[] = mul %5 %48
                %51:f64[] = add %49 %50
                %52:f64[] = add %45 %51
                %53:f64[] = mul %18 %1
                %54:f64[] = mul %0 %48
                %55:f64[] = add %53 %54
                %56:f64[] = mul %21 %1
                %57:f64[] = mul %0 %55
                %58:f64[] = add %56 %57
                %59:f64[] = add %52 %58
                %60:f64[] = mul %21 %1
                %61:f64[] = mul %0 %55
                %62:f64[] = add %60 %61
                %63:f64[] = add %59 %62
            in (%63)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_fourth_derivative_jit_rendering() {
    let mut context = ();
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(&mut context, fourth_derivative, 2.0f64).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = const
                %2:f64[] = const
                %3:f64[] = const
                %4:f64[] = const
                %5:f64[] = const
                %6:f64[] = const
                %7:f64[] = const
                %8:f64[] = const
                %9:f64[] = const
                %10:f64[] = const
                %11:f64[] = const
                %12:f64[] = const
                %13:f64[] = const
                %14:f64[] = const
                %15:f64[] = const
                %16:f64[] = mul %0 %0
                %17:f64[] = mul %16 %0
                %18:f64[] = mul %17 %0
                %19:f64[] = sin %0
                %20:f64[] = cos %0
                %21:f64[] = cos %0
                %22:f64[] = sin %0
                %23:f64[] = cos %0
                %24:f64[] = sin %0
                %25:f64[] = sin %0
                %26:f64[] = cos %0
                %27:f64[] = cos %0
                %28:f64[] = sin %0
                %29:f64[] = sin %0
                %30:f64[] = cos %0
                %31:f64[] = sin %0
                %32:f64[] = cos %0
                %33:f64[] = cos %0
                %34:f64[] = sin %0
                %35:f64[] = add %18 %19
                %36:f64[] = const
                %37:f64[] = const
                %38:f64[] = const
                %39:f64[] = const
                %40:f64[] = const
                %41:f64[] = const
                %42:f64[] = const
                %43:f64[] = const
                %44:f64[] = mul %27 %36
                %45:f64[] = mul %17 %36
                %46:f64[] = add %44 %45
                %47:f64[] = mul %0 %36
                %48:f64[] = mul %16 %47
                %49:f64[] = add %46 %48
                %50:f64[] = mul %0 %47
                %51:f64[] = mul %0 %50
                %52:f64[] = add %49 %51
                %53:f64[] = mul %0 %50
                %54:f64[] = add %52 %53
                %55:f64[] = const
                %56:f64[] = const
                %57:f64[] = const
                %58:f64[] = const
                %59:f64[] = mul %0 %55
                %60:f64[] = mul %50 %55
                %61:f64[] = mul %0 %55
                %62:f64[] = add %59 %61
                %63:f64[] = mul %50 %55
                %64:f64[] = add %60 %63
                %65:f64[] = mul %0 %62
                %66:f64[] = mul %47 %62
                %67:f64[] = add %64 %66
                %68:f64[] = mul %16 %55
                %69:f64[] = add %65 %68
                %70:f64[] = mul %47 %55
                %71:f64[] = mul %0 %69
                %72:f64[] = mul %36 %69
                %73:f64[] = add %67 %72
                %74:f64[] = mul %17 %55
                %75:f64[] = add %71 %74
                %76:f64[] = mul %36 %55
                %77:f64[] = mul %27 %55
                %78:f64[] = add %75 %77
                %79:f64[] = mul %36 %55
                %80:f64[] = neg %79
                %81:f64[] = mul %31 %80
                %82:f64[] = add %73 %81
                %83:f64[] = mul %16 %76
                %84:f64[] = add %82 %83
                %85:f64[] = mul %0 %76
                %86:f64[] = add %70 %85
                %87:f64[] = mul %0 %86
                %88:f64[] = add %84 %87
                %89:f64[] = mul %0 %86
                %90:f64[] = add %88 %89
                %91:f64[] = const
                %92:f64[] = const
                %93:f64[] = mul %0 %91
                %94:f64[] = mul %86 %91
                %95:f64[] = mul %0 %91
                %96:f64[] = add %93 %95
                %97:f64[] = mul %86 %91
                %98:f64[] = add %94 %97
                %99:f64[] = mul %0 %96
                %100:f64[] = mul %76 %96
                %101:f64[] = add %98 %100
                %102:f64[] = mul %16 %91
                %103:f64[] = add %99 %102
                %104:f64[] = mul %76 %91
                %105:f64[] = mul %31 %91
                %106:f64[] = mul %80 %91
                %107:f64[] = neg %105
                %108:f64[] = mul %36 %107
                %109:f64[] = mul %55 %107
                %110:f64[] = mul %36 %103
                %111:f64[] = add %108 %110
                %112:f64[] = mul %55 %103
                %113:f64[] = add %109 %112
                %114:f64[] = mul %36 %91
                %115:f64[] = mul %69 %91
                %116:f64[] = add %113 %115
                %117:f64[] = mul %47 %96
                %118:f64[] = add %111 %117
                %119:f64[] = mul %55 %96
                %120:f64[] = mul %16 %114
                %121:f64[] = add %118 %120
                %122:f64[] = mul %55 %114
                %123:f64[] = add %104 %122
                %124:f64[] = mul %47 %91
                %125:f64[] = mul %62 %91
                %126:f64[] = add %119 %125
                %127:f64[] = mul %0 %114
                %128:f64[] = add %124 %127
                %129:f64[] = mul %62 %114
                %130:f64[] = add %101 %129
                %131:f64[] = mul %50 %91
                %132:f64[] = add %121 %131
                %133:f64[] = mul %55 %91
                %134:f64[] = mul %0 %128
                %135:f64[] = add %132 %134
                %136:f64[] = mul %55 %128
                %137:f64[] = add %130 %136
                %138:f64[] = mul %50 %91
                %139:f64[] = add %135 %138
                %140:f64[] = mul %55 %91
                %141:f64[] = add %133 %140
                %142:f64[] = mul %0 %128
                %143:f64[] = add %139 %142
                %144:f64[] = mul %55 %128
                %145:f64[] = add %137 %144
                %146:f64[] = mul %0 %141
                %147:f64[] = add %126 %146
                %148:f64[] = mul %47 %141
                %149:f64[] = add %145 %148
                %150:f64[] = mul %0 %147
                %151:f64[] = add %116 %150
                %152:f64[] = mul %36 %147
                %153:f64[] = add %149 %152
                %154:f64[] = mul %33 %106
                %155:f64[] = add %153 %154
                %156:f64[] = mul %0 %123
                %157:f64[] = add %155 %156
                %158:f64[] = mul %0 %123
                %159:f64[] = add %157 %158
                %160:f64[] = const
                %161:f64[] = mul %0 %160
                %162:f64[] = mul %123 %160
                %163:f64[] = mul %0 %160
                %164:f64[] = add %161 %163
                %165:f64[] = mul %123 %160
                %166:f64[] = add %162 %165
                %167:f64[] = mul %33 %160
                %168:f64[] = mul %106 %160
                %169:f64[] = mul %36 %160
                %170:f64[] = mul %147 %160
                %171:f64[] = mul %47 %160
                %172:f64[] = mul %141 %160
                %173:f64[] = mul %0 %169
                %174:f64[] = add %171 %173
                %175:f64[] = mul %141 %169
                %176:f64[] = add %166 %175
                %177:f64[] = mul %55 %160
                %178:f64[] = mul %128 %160
                %179:f64[] = mul %55 %174
                %180:f64[] = mul %91 %174
                %181:f64[] = add %178 %180
                %182:f64[] = mul %55 %160
                %183:f64[] = add %177 %182
                %184:f64[] = mul %128 %160
                %185:f64[] = add %181 %184
                %186:f64[] = mul %55 %174
                %187:f64[] = add %179 %186
                %188:f64[] = mul %91 %174
                %189:f64[] = add %185 %188
                %190:f64[] = mul %62 %160
                %191:f64[] = mul %114 %160
                %192:f64[] = mul %0 %183
                %193:f64[] = add %190 %192
                %194:f64[] = mul %114 %183
                %195:f64[] = add %176 %194
                %196:f64[] = mul %62 %169
                %197:f64[] = add %187 %196
                %198:f64[] = mul %91 %169
                %199:f64[] = add %191 %198
                %200:f64[] = mul %47 %183
                %201:f64[] = add %197 %200
                %202:f64[] = mul %91 %183
                %203:f64[] = add %172 %202
                %204:f64[] = mul %55 %164
                %205:f64[] = add %193 %204
                %206:f64[] = mul %114 %164
                %207:f64[] = add %189 %206
                %208:f64[] = mul %55 %169
                %209:f64[] = mul %96 %169
                %210:f64[] = add %207 %209
                %211:f64[] = mul %36 %205
                %212:f64[] = add %201 %211
                %213:f64[] = mul %91 %205
                %214:f64[] = add %170 %213
                %215:f64[] = mul %80 %167
                %216:f64[] = add %212 %215
                %217:f64[] = mul %91 %167
                %218:f64[] = mul %76 %164
                %219:f64[] = add %216 %218
                %220:f64[] = mul %91 %164
                %221:f64[] = mul %76 %160
                %222:f64[] = add %208 %221
                %223:f64[] = mul %96 %160
                %224:f64[] = add %220 %223
                %225:f64[] = mul %86 %160
                %226:f64[] = add %219 %225
                %227:f64[] = mul %91 %160
                %228:f64[] = mul %0 %222
                %229:f64[] = add %226 %228
                %230:f64[] = mul %91 %222
                %231:f64[] = add %195 %230
                %232:f64[] = mul %86 %160
                %233:f64[] = add %229 %232
                %234:f64[] = mul %91 %160
                %235:f64[] = add %227 %234
                %236:f64[] = mul %0 %222
                %237:f64[] = add %233 %236
                %238:f64[] = mul %91 %222
                %239:f64[] = add %231 %238
                %240:f64[] = mul %0 %235
                %241:f64[] = add %224 %240
                %242:f64[] = mul %76 %235
                %243:f64[] = add %239 %242
                %244:f64[] = neg %217
                %245:f64[] = mul %36 %244
                %246:f64[] = add %210 %245
                %247:f64[] = mul %55 %244
                %248:f64[] = add %214 %247
                %249:f64[] = mul %36 %241
                %250:f64[] = add %246 %249
                %251:f64[] = mul %55 %241
                %252:f64[] = add %248 %251
                %253:f64[] = mul %47 %235
                %254:f64[] = add %250 %253
                %255:f64[] = mul %55 %235
                %256:f64[] = add %203 %255
                %257:f64[] = mul %0 %199
                %258:f64[] = add %254 %257
                %259:f64[] = mul %55 %199
                %260:f64[] = add %243 %259
                %261:f64[] = mul %0 %199
                %262:f64[] = add %258 %261
                %263:f64[] = mul %55 %199
                %264:f64[] = add %260 %263
                %265:f64[] = mul %0 %256
                %266:f64[] = add %252 %265
                %267:f64[] = mul %36 %256
                %268:f64[] = add %264 %267
                %269:f64[] = neg %168
                %270:f64[] = mul %34 %269
                %271:f64[] = add %268 %270
            in (%271)
        "}
        .trim_end(),
    );
}
pub(crate) fn assert_inline_fourth_derivative_jit_rendering() {
    let mut context = ();
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(
        &mut context,
        |context, x| {
            grad(
                context,
                |context, x| {
                    grad(
                        context,
                        |context, x| grad(context, quartic_plus_sin, x).expect("innermost grad should succeed"),
                        x,
                    )
                    .expect("third derivative should succeed")
                },
                x,
            )
            .expect("second derivative should succeed")
        },
        2.0f64,
    )
    .unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = const
                %2:f64[] = const
                %3:f64[] = const
                %4:f64[] = const
                %5:f64[] = const
                %6:f64[] = const
                %7:f64[] = const
                %8:f64[] = mul %0 %0
                %9:f64[] = mul %8 %0
                %10:f64[] = mul %9 %0
                %11:f64[] = sin %0
                %12:f64[] = cos %0
                %13:f64[] = cos %0
                %14:f64[] = sin %0
                %15:f64[] = cos %0
                %16:f64[] = sin %0
                %17:f64[] = sin %0
                %18:f64[] = cos %0
                %19:f64[] = add %10 %11
                %20:f64[] = const
                %21:f64[] = const
                %22:f64[] = const
                %23:f64[] = const
                %24:f64[] = mul %15 %20
                %25:f64[] = mul %9 %20
                %26:f64[] = add %24 %25
                %27:f64[] = mul %0 %20
                %28:f64[] = mul %8 %27
                %29:f64[] = add %26 %28
                %30:f64[] = mul %0 %27
                %31:f64[] = mul %0 %30
                %32:f64[] = add %29 %31
                %33:f64[] = mul %0 %30
                %34:f64[] = add %32 %33
                %35:f64[] = const
                %36:f64[] = const
                %37:f64[] = mul %0 %35
                %38:f64[] = mul %30 %35
                %39:f64[] = mul %0 %35
                %40:f64[] = add %37 %39
                %41:f64[] = mul %30 %35
                %42:f64[] = add %38 %41
                %43:f64[] = mul %0 %40
                %44:f64[] = mul %27 %40
                %45:f64[] = add %42 %44
                %46:f64[] = mul %8 %35
                %47:f64[] = add %43 %46
                %48:f64[] = mul %27 %35
                %49:f64[] = mul %0 %47
                %50:f64[] = mul %20 %47
                %51:f64[] = add %45 %50
                %52:f64[] = mul %9 %35
                %53:f64[] = add %49 %52
                %54:f64[] = mul %20 %35
                %55:f64[] = mul %15 %35
                %56:f64[] = add %53 %55
                %57:f64[] = mul %20 %35
                %58:f64[] = neg %57
                %59:f64[] = mul %17 %58
                %60:f64[] = add %51 %59
                %61:f64[] = mul %8 %54
                %62:f64[] = add %60 %61
                %63:f64[] = mul %0 %54
                %64:f64[] = add %48 %63
                %65:f64[] = mul %0 %64
                %66:f64[] = add %62 %65
                %67:f64[] = mul %0 %64
                %68:f64[] = add %66 %67
                %69:f64[] = const
                %70:f64[] = mul %0 %69
                %71:f64[] = mul %64 %69
                %72:f64[] = mul %0 %69
                %73:f64[] = add %70 %72
                %74:f64[] = mul %64 %69
                %75:f64[] = add %71 %74
                %76:f64[] = mul %0 %73
                %77:f64[] = mul %54 %73
                %78:f64[] = add %75 %77
                %79:f64[] = mul %8 %69
                %80:f64[] = add %76 %79
                %81:f64[] = mul %54 %69
                %82:f64[] = mul %17 %69
                %83:f64[] = mul %58 %69
                %84:f64[] = neg %82
                %85:f64[] = mul %20 %84
                %86:f64[] = mul %35 %84
                %87:f64[] = mul %20 %80
                %88:f64[] = add %85 %87
                %89:f64[] = mul %35 %80
                %90:f64[] = add %86 %89
                %91:f64[] = mul %20 %69
                %92:f64[] = mul %47 %69
                %93:f64[] = add %90 %92
                %94:f64[] = mul %27 %73
                %95:f64[] = add %88 %94
                %96:f64[] = mul %35 %73
                %97:f64[] = mul %8 %91
                %98:f64[] = add %95 %97
                %99:f64[] = mul %35 %91
                %100:f64[] = add %81 %99
                %101:f64[] = mul %27 %69
                %102:f64[] = mul %40 %69
                %103:f64[] = add %96 %102
                %104:f64[] = mul %0 %91
                %105:f64[] = add %101 %104
                %106:f64[] = mul %40 %91
                %107:f64[] = add %78 %106
                %108:f64[] = mul %30 %69
                %109:f64[] = add %98 %108
                %110:f64[] = mul %35 %69
                %111:f64[] = mul %0 %105
                %112:f64[] = add %109 %111
                %113:f64[] = mul %35 %105
                %114:f64[] = add %107 %113
                %115:f64[] = mul %30 %69
                %116:f64[] = add %112 %115
                %117:f64[] = mul %35 %69
                %118:f64[] = add %110 %117
                %119:f64[] = mul %0 %105
                %120:f64[] = add %116 %119
                %121:f64[] = mul %35 %105
                %122:f64[] = add %114 %121
                %123:f64[] = mul %0 %118
                %124:f64[] = add %103 %123
                %125:f64[] = mul %27 %118
                %126:f64[] = add %122 %125
                %127:f64[] = mul %0 %124
                %128:f64[] = add %93 %127
                %129:f64[] = mul %20 %124
                %130:f64[] = add %126 %129
                %131:f64[] = mul %18 %83
                %132:f64[] = add %130 %131
                %133:f64[] = mul %0 %100
                %134:f64[] = add %132 %133
                %135:f64[] = mul %0 %100
                %136:f64[] = add %134 %135
            in (%136)
        "}
        .trim_end(),
    );
}

#[cfg(any(feature = "ndarray", test))]
use ndarray::{Array2, arr2};

#[cfg(any(feature = "ndarray", test))]
fn bilinear_matmul<Context, M>(_: &mut Context, inputs: (M, M)) -> M
where
    M: MatrixOps,
{
    inputs.0.matmul(inputs.1)
}

#[cfg(any(feature = "ndarray", test))]
fn three_matmul_sine<Context, M>(_: &mut Context, inputs: (M, M, M, M)) -> M
where
    M: MatrixOps + FloatExt,
{
    let (x, a, b, c) = inputs;
    x.matmul(a).sin().matmul(b).matmul(c)
}

#[cfg(any(feature = "ndarray", test))]
fn first_matrix_gradient<Context, V>(context: &mut Context, inputs: (V, V, V, V)) -> V
where
    V: MatrixValue
        + OneLike
        + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<Linearized<V>, To = Linearized<V>>,
{
    let (x_bar, _, _, _) = grad(context, three_matmul_sine, inputs).expect("matrix gradient should succeed");
    x_bar
}

#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_jit_rendering() {
    let mut context = ();
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);
    let (_, compiled): (Array2<f64>, CompiledFunction<Array2<f64>, (Array2<f64>, Array2<f64>), Array2<f64>>) =
        jit(&mut context, bilinear_matmul, (a, b)).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[2,2], %1:f64[2,2] .
            let %2:f64[2,2] = matmul %0 %1
            in (%2)
        "}
        .trim_end(),
    );
}

#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_pushforward_rendering() {
    let mut context = ();
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);
    let (_, pushforward): (Array2<f64>, LinearProgram<Array2<f64>, (Array2<f64>, Array2<f64>), Array2<f64>>) =
        linearize(&mut context, bilinear_matmul, (a, b)).unwrap();

    assert_eq!(
        pushforward.to_string(),
        indoc! {"
            lambda %0:f64[2,2], %1:f64[2,2] .
            let %2:f64[2,2] = right_matmul %0
                %3:f64[2,2] = left_matmul %1
                %4:f64[2,2] = add %2 %3
            in (%4)
        "}
        .trim_end(),
    );
}

#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_pullback_rendering() {
    let mut context = ();
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);
    let (_, pullback): (Array2<f64>, LinearProgram<Array2<f64>, Array2<f64>, (Array2<f64>, Array2<f64>)>) =
        vjp(&mut context, bilinear_matmul, (a, b)).unwrap();

    assert_eq!(
        pullback.to_string(),
        indoc! {"
            lambda %0:f64[2,2] .
            let %1:f64[2,2] = left_matmul %0
                %2:f64[2,2] = right_matmul %0
                %3:f64[2,2] = const
            in (%2, %1)
        "}
        .trim_end(),
    );
}
#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_hessian_style_jit_rendering() {
    let mut context = ();
    let x = arr2(&[[0.7f64]]);
    let a = arr2(&[[2.0f64]]);
    let b = arr2(&[[-1.5f64]]);
    let c = arr2(&[[4.0f64]]);
    let (_, compiled): (
        Array2<f64>,
        CompiledFunction<Array2<f64>, (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), Array2<f64>>,
    ) = jit(
        &mut context,
        |context, inputs| {
            let seeds = (inputs.0.one_like(), inputs.1.zero_like(), inputs.2.zero_like(), inputs.3.zero_like());
            jvp(context, first_matrix_gradient, inputs, seeds).expect("matrix Hessian should succeed").1
        },
        (x, a, b, c),
    )
    .unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[1,1], %1:f64[1,1], %2:f64[1,1], %3:f64[1,1] .
            let %4:f64[1,1] = const
                %5:f64[1,1] = const
                %6:f64[1,1] = const
                %7:f64[1,1] = const
                %8:f64[1,1] = const
                %9:f64[1,1] = const
                %10:f64[1,1] = const
                %11:f64[1,1] = matmul %0 %1
                %12:f64[1,1] = sin %11
                %13:f64[1,1] = cos %11
                %14:f64[1,1] = cos %11
                %15:f64[1,1] = sin %11
                %16:f64[1,1] = matmul %12 %2
                %17:f64[1,1] = matmul %16 %3
                %18:f64[1,1] = matrix_transpose %16
                %19:f64[1,1] = matrix_transpose %3
                %20:f64[1,1] = matrix_transpose %12
                %21:f64[1,1] = matrix_transpose %2
                %22:f64[1,1] = matrix_transpose %0
                %23:f64[1,1] = matrix_transpose %1
                %24:f64[1,1] = const
                %25:f64[1,1] = const
                %26:f64[1,1] = matmul %18 %24
                %27:f64[1,1] = matmul %24 %19
                %28:f64[1,1] = matmul %20 %27
                %29:f64[1,1] = matmul %27 %21
                %30:f64[1,1] = mul %14 %29
                %31:f64[1,1] = matmul %22 %30
                %32:f64[1,1] = matmul %30 %23
                %33:f64[1,1] = matmul %4 %1
                %34:f64[1,1] = matmul %0 %5
                %35:f64[1,1] = add %33 %34
                %36:f64[1,1] = mul %13 %35
                %37:f64[1,1] = mul %15 %35
                %38:f64[1,1] = neg %37
                %39:f64[1,1] = matmul %36 %2
                %40:f64[1,1] = matmul %12 %6
                %41:f64[1,1] = add %39 %40
                %42:f64[1,1] = matmul %41 %3
                %43:f64[1,1] = matmul %16 %7
                %44:f64[1,1] = add %42 %43
                %45:f64[1,1] = matrix_transpose %41
                %46:f64[1,1] = matrix_transpose %7
                %47:f64[1,1] = matrix_transpose %36
                %48:f64[1,1] = matrix_transpose %6
                %49:f64[1,1] = matrix_transpose %4
                %50:f64[1,1] = matrix_transpose %5
                %51:f64[1,1] = matmul %45 %24
                %52:f64[1,1] = matmul %18 %25
                %53:f64[1,1] = add %51 %52
                %54:f64[1,1] = matmul %25 %19
                %55:f64[1,1] = matmul %24 %46
                %56:f64[1,1] = add %54 %55
                %57:f64[1,1] = matmul %47 %27
                %58:f64[1,1] = matmul %20 %56
                %59:f64[1,1] = add %57 %58
                %60:f64[1,1] = matmul %56 %21
                %61:f64[1,1] = matmul %27 %48
                %62:f64[1,1] = add %60 %61
                %63:f64[1,1] = mul %29 %38
                %64:f64[1,1] = mul %14 %62
                %65:f64[1,1] = add %63 %64
                %66:f64[1,1] = matmul %49 %30
                %67:f64[1,1] = matmul %22 %65
                %68:f64[1,1] = add %66 %67
                %69:f64[1,1] = matmul %65 %23
                %70:f64[1,1] = matmul %30 %50
                %71:f64[1,1] = add %69 %70
            in (%71)
        "}
        .trim_end(),
    );
}
