import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

CASOS_EJEMPLO = [
    {
        "Nombre": "Luis Orihuela",
        "Edad": "23 (Adulto Joven)",
        "Facultad": "FIA",
        "Procedencia": "Lima",
        "Vivienda": "Residencia",
        "TitulacionMadre": "Técnica",
        "TitulacionPadre": "Educación Superior",
        "OcupacionPadre": "Labora",
        "OcupacionMadre": "Labora",
        "Trasladado": "No",
        "PromedioPonderado": "Excelente",
        "DeudaFinanciera": "Sin deuda",
        "SintomasDepresion": "Bajo",
        "SituacionLaboral": "Medio tiempo",
        "Asistencia": "Alta",
        "DesarrolloTareas": "Alto",
        "ProblemasEspeciales": "No",
        "NivelRiesgoReal": "Bajo"
    },
    {
        "Nombre": "Brayan Ponce",
        "Edad": "18 (Joven)",
        "Facultad": "Ciencias Salud",
        "Procedencia": "Lima",
        "Vivienda": "Alquila cuarto",
        "TitulacionMadre": "Secundaria",
        "TitulacionPadre": "Secundaria",
        "OcupacionPadre": "Labora",
        "OcupacionMadre": "Labora",
        "Trasladado": "Sí",
        "PromedioPonderado": "Bajo",
        "DeudaFinanciera": "Deuda alta",
        "SintomasDepresion": "Medio",
        "SituacionLaboral": "No trabaja",
        "Asistencia": "Muy baja",
        "DesarrolloTareas": "Bajo",
        "ProblemasEspeciales": "Sí",
        "NivelRiesgoReal": "Alto"
    },
    {
        "Nombre": "Carlos Acosta",
        "Edad": "25 (Adulto Joven)",
        "Facultad": "Teología",
        "Procedencia": "Lima",
        "Vivienda": "Familiares",
        "TitulacionMadre": "Secundaria",
        "TitulacionPadre": "Secundaria",
        "OcupacionPadre": "Labora",
        "OcupacionMadre": "Ama de casa",
        "Trasladado": "No",
        "PromedioPonderado": "Excelente",
        "DeudaFinanciera": "Deuda baja",
        "SintomasDepresion": "Medio",
        "SituacionLaboral": "Mediotiempo",
        "Asistencia": "Medio",
        "DesarrolloTareas": "Medio",
        "ProblemasEspeciales": "No",
        "NivelRiesgoReal": "Medio"
    },
    {
        "Nombre": "Aylan Mostacero",
        "Edad": "27 (Adulto)",
        "Facultad": "Teología",
        "Procedencia": "Provincia",
        "Vivienda": "Familiares",
        "TitulacionMadre": "Secundaria",
        "TitulacionPadre": "Secundaria",
        "OcupacionPadre": "Desempleado",
        "OcupacionMadre": "Ama de casa",
        "Trasladado": "No",
        "PromedioPonderado": "Bueno",
        "DeudaFinanciera": "Deuda media",
        "SintomasDepresion": "Medio",
        "SituacionLaboral": "Tiempo completo",
        "Asistencia": "Bajo",
        "DesarrolloTareas": "Bajo",
        "ProblemasEspeciales": "No",
        "NivelRiesgoReal": "Alto"
    },
    {
        "Nombre": "Esau Morales",
        "Edad": "30 (Adulto)",
        "Facultad": "Ciencias Salud",
        "Procedencia": "Provincia",
        "Vivienda": "Familiares",
        "TitulacionMadre": "Secundaria",
        "TitulacionPadre": "Secundaria",
        "OcupacionPadre": "Desempleado",
        "OcupacionMadre": "Ama de casa",
        "Trasladado": "Sí",
        "PromedioPonderado": "Bueno",
        "DeudaFinanciera": "Deuda media",
        "SintomasDepresion": "Bajo",
        "SituacionLaboral": "No trabaja",
        "Asistencia": "Medio",
        "DesarrolloTareas": "Alto",
        "ProblemasEspeciales": "No",
        "NivelRiesgoReal": "Medio"
    }
]


# ==========================
# Configuración global
# ==========================
st.set_page_config(
    page_title="Riesgo de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artefactos"
PIPE_PATH = MODEL_DIR / "pipeline_mlp_ohe.joblib"

FEATURES = [
    "PromedioPonderado",
    "DeudaFinanciera",
    "SintomasDepresion",
    "Edad",
    "SituacionLaboral",
    "Trasladado",
    "Asistencia",
]

CLASSES = ["Bajo", "Medio", "Alta"]

DOMS = {
    "PromedioPonderado": ["Bajo", "Bueno", "Excelente"],
    "DeudaFinanciera": ["Sindeuda", "Deudabaja", "Deudamedia", "Deudaalta"],
    "SintomasDepresion": ["Bajo", "Medio", "Alto"],
    "Edad": ["Joven", "Adultojoven", "Adulto", "Adultomayor"],
    "SituacionLaboral": ["Notrabaja", "Mediotiempo", "Tiempocompleto"],
    "Trasladado": ["No", "Si"],
    "Asistencia": ["Muybajo", "Bajo", "Medio", "Alta"],
}

# ==========================
# Cargar modelo
# ==========================
try:
    pipe = joblib.load(PIPE_PATH)
except Exception as e:
    st.error(f"Error cargando el modelo desde {PIPE_PATH}:\n{e}")
    st.stop()

# ==========================
# Helpers
# ==========================
def _norm_value(v: str) -> str:
    return str(v).strip().replace("\n", " ").title()

def _norm_row(row: dict) -> dict:
    return {f: _norm_value(row[f]) for f in FEATURES}

def predecir(row: dict):
    df = pd.DataFrame([row], columns=FEATURES)
    proba = pipe.predict_proba(df)[0]

    if hasattr(pipe, "classes_"):
        order = np.argsort(pipe.classes_)
        proba = proba[order]

    idx = int(np.argmax(proba))
    return idx, proba

# ==========================
# Vista Redes Neuronales
# ==========================
def vista_redes():
    # Estado
    if "show_form" not in st.session_state:
        st.session_state["show_form"] = False
    if "records" not in st.session_state:
        st.session_state["records"] = []

    # Header
    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.title("🎓 Predicción de riesgo de deserción")
    with header_right:
        if st.button("➕ Simular Caso", use_container_width=True):
            st.session_state["show_form"] = True

    st.write("Demo usando tu modelo MLP entrenado.")

    # Formulario
    if st.session_state["show_form"]:
        st.markdown("### Datos del estudiante")
        with st.form("form_estudiante"):
            cols = st.columns(2)
            inputs = {}
            for i, feat in enumerate(FEATURES):
                opts = DOMS[feat]
                col = cols[i % 2]
                inputs[feat] = col.selectbox(feat, opts, index=0)
            submitted = st.form_submit_button("Calcular riesgo")

        if submitted:
            row_norm = _norm_row(inputs)
            idx, proba = predecir(row_norm)

            st.markdown("### Resultado")
            st.metric("Riesgo predicho", CLASSES[idx])

            prob_pred = float(proba[idx]) * 100.0
            st.write(f"Probabilidad de esa clase: **{prob_pred:.2f}%**")

            row_hist = row_norm.copy()
            row_hist["RiesgoPredicho"] = CLASSES[idx]
            row_hist["Probabilidad(%)"] = round(prob_pred, 2)
            st.session_state["records"].append(row_hist)
    else:
        st.info("Pulsa **Agregar estudiante** para registrar un nuevo caso.")
        
    # ==========================
    # Historial de Casos Sintéticos
    # ==========================

    st.markdown("---")
    st.markdown("### 📋 Historial de casos simulados")

    df_casos = pd.DataFrame(CASOS_EJEMPLO)
    st.dataframe(df_casos, use_container_width=True, hide_index=True)

# selector de caso (simula el ícono de ojo)
    nombres = ["-- Seleccione un estudiante --"] + df_casos["Nombre"].tolist()
    seleccion = st.selectbox("👁️ Ver detalle de un caso", nombres, index=0)

# Solo mostrar detalle si eligió un estudiante real
    if seleccion != "-- Seleccione un estudiante --":
        caso_sel = df_casos[df_casos["Nombre"] == seleccion].iloc[0]

        st.markdown(f"#### 🧑‍🎓 Perfil de {caso_sel['Nombre']}")

    # ===== CSS común para todos los casos =====
        st.markdown(
        """
        <style>
        .card {
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            color: #111827;
            box-shadow: 0 10px 8px rgba(15, 23, 42, 0.08);
        }
        .card h5 {
            margin: 0 0 0.4rem 0;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .pill {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            margin: 0.1rem 0.25rem 0.1rem 0;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 0.8rem;
            font-weight: 600;
        }
        /* Timeline tipo stepper */
        .timeline {
            position: relative;
            margin-left: 0;
            padding-left: 0;
        }
        .timeline-item {
            position: relative;
            padding: 1rem 0 1rem 3.5rem;
            font-size: 0.95rem;
        }
        .timeline-circle {
            position: absolute;
            left: 0;
            top: 1rem;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #1e40af;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.1rem;
            box-shadow: 0 2px 8px rgba(30, 64, 175, 0.3);
            z-index: 2;
        }
        .timeline-item:not(:last-child)::after {
            content: "";
            position: absolute;
            left: 19px;
            top: 50px;
            width: 2px;
            height: calc(100% - 30px);
            background: #cbd5e1;
            z-index: 1;
        }
        .timeline-day {
            font-weight: 700;
            color: #111827;
            display: block;
            margin-bottom: 0.2rem;
        }
        .timeline-content {
            color: #4b5563;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ===== Barra de riesgo dinámica según NivelRiesgoReal =====
        nivel_riesgo = caso_sel["NivelRiesgoReal"]

        if nivel_riesgo == "Bajo":
            color_barra = "#22c55e"
            ancho_barra = "30%"
            badge_bg = "#dcfce7"
            badge_text = "#16a34a"
        elif nivel_riesgo == "Medio":
            color_barra = "#f59e0b"
            ancho_barra = "60%"
            badge_bg = "#fef3c7"
            badge_text = "#d97706"
        else:  # Alto
            color_barra = "#ef4444"
            ancho_barra = "90%"
            badge_bg = "#fee2e2"
            badge_text = "#dc2626"

        st.markdown(
        f"""
        <div class="card">
          <h5>Resumen de nivel de riesgo</h5>
          <div style="background:#e5e7eb; border-radius:999px; height:10px; overflow:hidden; margin-bottom:0.5rem;">
            <div style="width:{ancho_barra}; background:{color_barra}; height:100%;"></div>
          </div>
          <span style="display:inline-block; padding:0.25rem 0.8rem; border-radius:999px; background:{badge_bg}; color:{badge_text}; font-size:0.85rem; font-weight:600;">
            Riesgo {nivel_riesgo.lower()}
          </span>
          <p style="margin-top:0.4rem; font-size:0.9rem; color:#4b5563;">
            Actualmente tu riesgo de deserción es <b>{nivel_riesgo.lower()}</b>. El objetivo es acompañarte con un plan ajustado a tu situación.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ===== CONTENIDO POR CASO =====

    # CASO 1: Luis Orihuela (tu contenido original)
        if caso_sel["Nombre"] == "Luis Orihuela":
        # Diagnóstico
            st.markdown(
            """
            <div class="card">
              <h5>Diagnóstico del estado actual</h5>
              <p>
              Estudiante de la Facultad FIA, con <b>promedio excelente</b>, 
              <b>asistencia alta</b> y <b>desarrollo de tareas alto</b>. 
              Trabaja a <b>medio tiempo</b>, es <b>adulto joven</b>, vive en <b>residencia</b> y no tiene <b>deuda financiera</b>.
              </p>
              <p>
              Los síntomas de depresión son <b>bajos</b> y no se reportan problemas especiales. 
              El sistema clasifica el riesgo de deserción como <b>bajo</b>.
              </p>
              <p>
              El principal reto no es académico, sino <b>mantener el equilibrio</b> entre trabajo, estudios y bienestar personal.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Fortalezas / Puntos a cuidar
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                """
                <div class="card">
                  <h5>Fortalezas principales</h5>
                  <ul>
                    <li>Rendimiento <b>excelente</b> y dominio de contenidos.</li>
                    <li><b>Desarrollo de tareas alto</b> y constante.</li>
                    <li><b>Asistencia alta</b> y buena disciplina.</li>
                    <li>Síntomas emocionales <b>bajos</b>.</li>
                    <li>Padres con formación técnica y superior → <b>apoyo académico</b>.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with col2:
                st.markdown(
                """
                <div class="card">
                  <h5>Puntos a cuidar</h5>
                  <ul>
                    <li>Trabajo de <b>medio tiempo</b> en épocas de alta carga.</li>
                    <li>Vivir en <b>residencia</b> puede reducir red emocional cercana.</li>
                    <li>Posibles <b>cambios económicos</b> futuros.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recomendaciones + Plan (como ya lo tenías)
            col_rec, col_plan = st.columns(2)
            with col_rec:
                st.markdown(
                """
                <div class="card">
                  <h5>Recomendaciones accionables</h5>
                  <span class="pill">Académicas</span>
                  <ul>
                    <li>Diseñar una <b>agenda semanal integrada</b> (clases, estudio, trabajo, descanso).</li>
                    <li>Priorizar cursos de <b>mayor peso y dificultad</b> con tiempo extra.</li>
                    <li>Aplicar <b>estudio activo</b>: resúmenes, ejercicios, explicar a otros.</li>
                  </ul>
                  <span class="pill">Emocionales / Motivacionales</span>
                  <ul>
                    <li>Crear <b>micro-rutinas de autocuidado</b> (pausas, caminatas, respiración).</li>
                    <li>Definir <b>señales tempranas de estrés</b> y cómo actuar.</li>
                    <li>Mantener un <b>registro de logros</b>.</li>
                  </ul>
                  <span class="pill">Administrativas</span>
                  <ul>
                    <li>Agendar <b>tutoría académica</b> en FIA.</li>
                    <li>Contactar al <b>servicio psicopedagógico</b>.</li>
                    <li>Informarte sobre <b>becas y apoyos económicos</b>.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with col_plan:
                st.markdown(
                """
                <div class="card">
                  <h5>Plan de acción: próximos 7 días</h5>
                  <div class="timeline">
                    <div class="timeline-item">
                      <div class="timeline-circle">1</div>
                      <span class="timeline-day">Día 1</span>
                      <span class="timeline-content">Diseñar tu horario semanal integrado.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">2</div>
                      <span class="timeline-day">Día 2</span>
                      <span class="timeline-content">Elegir 2–3 cursos más exigentes y darles tiempo extra.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">3</div>
                      <span class="timeline-day">Día 3</span>
                      <span class="timeline-content">Escribir tus señales tempranas de estrés y tus respuestas.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">4</div>
                      <span class="timeline-day">Día 4</span>
                      <span class="timeline-content">Contactar a tu tutor/a o coordinador/a.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">5</div>
                      <span class="timeline-day">Día 5</span>
                      <span class="timeline-content">Agendar una cita psicopedagógica.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">6</div>
                      <span class="timeline-day">Día 6</span>
                      <span class="timeline-content">Crear tu registro de logros.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">7</div>
                      <span class="timeline-day">Día 7</span>
                      <span class="timeline-content">Revisar la semana y ajustar tu plan.</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Mensaje final
            st.markdown(
            """
            <div class="card">
              <h5>Mensaje final</h5>
              <p>
              Tu perfil muestra a una persona <b>responsable</b>, <b>constante</b> y con muy buen desempeño académico.
              </p>
              <p>
              No se trata de hacer más, sino de <b>organizar mejor</b> lo que ya haces bien y apoyarte en la red institucional disponible.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # CASO 2: Brayan Ponce (usando el texto que me enviaste)
        elif caso_sel["Nombre"] == "Brayan Ponce":
        # Diagnóstico
            st.markdown(
            """
            <div class="card">
              <h5>Diagnóstico del estado actual</h5>
              <p>
              Estudiante de 18 años, de <b>Ciencias de la Salud</b>, con promedio <b>bajo</b>, 
              <b>asistencia muy baja</b>, desarrollo de tareas <b>bajo</b>, síntomas de depresión nivel <b>medio</b>, 
              <b>deuda financiera alta</b> y viviendo en <b>cuarto alquilado</b> en Lima.
              </p>
              <p><b>Señales de alerta principales:</b></p>
              <ul>
                <li>Asistencia <b>muy baja</b> a clases (crítico en carreras de salud).</li>
                <li>Promedio y desarrollo de tareas <b>bajos</b>.</li>
                <li><b>Deuda financiera alta</b> y contexto económico ajustado.</li>
                <li>Síntomas de depresión en nivel <b>medio</b>.</li>
                <li>Traslado reciente y vivienda alquilada → menor red de soporte.</li>
                <li>Reporte de <b>problemas especiales</b> que añaden presión.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Fortalezas / Debilidades
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                """
                <div class="card">
                  <h5>Fortalezas actuales</h5>
                  <ul>
                    <li><b>18 años</b>: alta capacidad de adaptación si hay apoyo.</li>
                    <li><b>No trabajas</b>: potencial para reorganizar tu tiempo.</li>
                    <li>Decisión de seguir en la carrera pese al traslado → <b>intención de continuar</b>.</li>
                    <li>Ambos padres <b>laboran</b>, posible apoyo económico básico.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with col2:
                st.markdown(
                """
                <div class="card">
                  <h5>Debilidades críticas</h5>
                  <ul>
                    <li><b>Bajo promedio + tareas bajas</b> → dificultades de autogestión y organización.</li>
                    <li><b>Asistencia muy baja</b> → afecta laboratorios y práctica.</li>
                    <li><b>Síntomas de depresión medio</b> → menos energía y motivación.</li>
                    <li><b>Deuda alta</b> → estrés y preocupación constante.</li>
                    <li>Vivienda en <b>cuarto alquilado</b> → más responsabilidades y posible aislamiento.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recomendaciones + Plan
            col_rec, col_plan = st.columns(2)
            with col_rec:
                st.markdown(
                """
                <div class="card">
                  <h5>Recomendaciones accionables</h5>
                  <span class="pill">Académicas</span>
                  <ul>
                    <li>Elegir <b>2–3 cursos prioritarios</b> para “rescate” este ciclo.</li>
                    <li>Reiniciar asistencia con meta realista: al menos <b>70%</b> en esos cursos.</li>
                    <li>Usar <b>técnica Pomodoro</b>: 25 min estudio + 5 min pausa, repetir 3 veces.</li>
                    <li>Crear una <b>micro-red académica</b> (1 compañero/a por curso).</li>
                  </ul>
                  <span class="pill">Emocionales / Motivacionales</span>
                  <ul>
                    <li><b>Pausa diagnóstica emocional</b> diaria de 10–15 minutos (preocupación, logro, acción).</li>
                    <li><b>Micro-hábitos</b>: dormir 7h, comer algo ligero antes de clase, moverse 10–15 min.</li>
                    <li><b>Normalizar pedir ayuda psicológica</b> y agendar cita.</li>
                    <li>Aplicar regla de <b>“solo 10 minutos”</b> cuando sientas bloqueo total.</li>
                  </ul>
                  <span class="pill">Administrativas</span>
                  <ul>
                    <li>Revisión de <b>deuda financiera</b> en finanzas (monto, plazos, fraccionamiento, becas).</li>
                    <li>Contactar <b>tutor académico</b> o coordinador para ajustar carga de cursos.</li>
                    <li>Acceder a <b>servicios institucionales</b>: tutorías, consejería, nivelación.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with col_plan:
                st.markdown(
                """
                <div class="card">
                  <h5>Plan de acción: próximos 7 días</h5>
                  <div class="timeline">
                    <div class="timeline-item">
                      <div class="timeline-circle">1</div>
                      <span class="timeline-day">Día 1</span>
                      <span class="timeline-content">Listar cursos y marcar 2–3 prioritarios. Revisar tareas y evaluaciones próximas.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">2</div>
                      <span class="timeline-day">Día 2</span>
                      <span class="timeline-content">Ir a finanzas o escribir correo para entender deuda y opciones. Crear lista de tareas de la semana.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">3</div>
                      <span class="timeline-day">Día 3</span>
                      <span class="timeline-content">Asistir a todas las clases de al menos 2 cursos prioritarios y hablar con un compañero por curso.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">4</div>
                      <span class="timeline-day">Día 4</span>
                      <span class="timeline-content">Hacer 2 bloques de estudio (60–90 min) con Pomodoro y pedir cita psicológica.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">5</div>
                      <span class="timeline-day">Día 5</span>
                      <span class="timeline-content">Asistir a la cita (o insistir) y preparar material para la siguiente semana.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">6</div>
                      <span class="timeline-day">Día 6</span>
                      <span class="timeline-content">Hacer una sesión de estudio en pareja/grupo y aplicar “solo 10 minutos” en alguna tarea evitada.</span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">7</div>
                      <span class="timeline-day">Día 7</span>
                      <span class="timeline-content">Evaluar la semana (asistencias, bloques de estudio) y ajustar el plan para la siguiente.</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Mensaje final
            st.markdown(
            """
            <div class="card">
              <h5>Mensaje final</h5>
              <p>
              Tu situación muestra <b>riesgo alto</b>, pero también muestra algo importante: sigues en la universidad y estás buscando ayuda.
              </p>
              <p>
              Con acciones concretas en lo académico, emocional y administrativo, es posible <b>reducir ese riesgo</b>.
              No necesitas hacerlo todo de golpe ni hacerlo solo/a: apóyate en los recursos de la universidad y tu entorno cercano.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        elif caso_sel["Nombre"] == "Carlos Acosta":
        # Diagnóstico del estado actual
            st.markdown(
            """
            <div class="card">
              <h5>Diagnóstico del estado actual</h5>
              <p>
              Estudiante de <b>Teología</b>, 25 años (adulto joven), con <b>promedio ponderado excelente</b>, 
              lo que indica muy buen desempeño académico global. Trabajas a <b>medio tiempo</b>, 
              con <b>asistencia media</b> y <b>desarrollo de tareas medio</b>.
              </p>
              <p><b>Señales de alerta:</b></p>
              <ul>
                <li>Riesgo de deserción <b>medio</b>, asociado a factores económicos, emocionales y familiares.</li>
                <li>Asistencia y tareas en nivel <b>medio</b>, por debajo de tu potencial (promedio excelente).</li>
              </ul>
              <p>
              En Teología se espera alta constancia en lectura, reflexión y participación. 
              Tu promedio indica capacidades fuertes, pero la asistencia media y el estado emocional 
              podrían afectar tu continuidad si no se atienden.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Fortalezas y debilidades
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                """
                <div class="card">
                  <h5>Fortalezas actuales</h5>
                  <ul>
                    <li><b>Promedio excelente</b>: alta comprensión y disciplina intelectual.</li>
                    <li><b>Adulto joven</b> con experiencia laboral → responsabilidad y organización.</li>
                    <li>Vives con <b>familiares</b> en Lima: cierto soporte económico y emocional.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with col2:
                st.markdown(
                """
                <div class="card">
                  <h5>Debilidades críticas (potenciales)</h5>
                  <ul>
                    <li><b>Síntomas de depresión medio</b> → afecta motivación y concentración.</li>
                    <li><b>Tareas y asistencia medias</b> → brecha entre tu potencial y tu práctica diaria.</li>
                    <li>Padres con <b>nivel secundario</b> → apoyo, pero quizá sin herramientas académicas.</li>
                  </ul>
                  <p>
                  El riesgo es <b>medio</b> no por falta de capacidad, sino por la combinación 
                  de estrés financiero moderado, trabajo + estudios y estado emocional.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recomendaciones y plan
            col_rec, col_plan = st.columns(2)

            with col_rec:
                st.markdown(
                """
                <div class="card">
                  <h5>Recomendaciones accionables</h5>
                  <span class="pill">Académicas</span>
                  <ul>
                    <li>Definir <b>3–4 bloques fijos</b> de estudio (60–90 min) compatibles con tu trabajo.</li>
                    <li>Priorizar cursos que fortalecen <b>pensamiento teológico, ético y de servicio</b>.</li>
                    <li>Aplicar <b>lectura activa</b>: subrayar, resumir (5–7 líneas) y formular 1–2 preguntas críticas.</li>
                    <li>Centralizar apuntes en <b>Notion / OneNote / Google Docs</b> por curso.</li>
                  </ul>
                  <span class="pill">Emocionales / Motivacionales</span>
                  <ul>
                    <li>Rutina diaria de <b>auto-revisión emocional</b> (5–10 min: ánimo 1–10, carga, gratitud).</li>
                    <li>Fragmentar tareas grandes en <b>pasos pequeños</b> (tema, esquema, borrador, revisión).</li>
                    <li>Incluir <b>pausas intencionales</b> entre trabajo y estudio (10–15 min sin pantallas).</li>
                    <li>Buscar al menos <b>una persona de confianza</b> para hablar de cómo te sientes.</li>
                  </ul>
                  <span class="pill">Administrativas</span>
                  <ul>
                    <li>Solicitar <b>tutoría académica</b> para ajustar carga de cursos y próximos ciclos.</li>
                    <li>Consultar en <b>bienestar universitario</b> por orientación psicológica o consejería.</li>
                    <li>Revisar en oficina administrativa el estado de <b>deuda</b> y opciones de fraccionamiento.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with col_plan:
                st.markdown(
                """
                <div class="card">
                  <h5>Plan de acción: próximos 7 días</h5>
                  <div class="timeline">
                    <div class="timeline-item">
                      <div class="timeline-circle">1</div>
                      <span class="timeline-day">Día 1</span>
                      <span class="timeline-content">
                        Organizar la semana: calendario con trabajo, clases y al menos 3 bloques de estudio.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">2</div>
                      <span class="timeline-day">Día 2</span>
                      <span class="timeline-content">
                        Aplicar lectura activa en un texto clave de Teología y registrar tu ánimo (1–10) y una gratitud.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">3</div>
                      <span class="timeline-day">Día 3</span>
                      <span class="timeline-content">
                        Dividir una tarea grande en subtareas con fechas (esquema, borrador, revisión) 
                        y hacer una pausa intencional entre trabajo y estudio.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">4</div>
                      <span class="timeline-day">Día 4</span>
                      <span class="timeline-content">
                        Agendar <b>tutoría académica</b> y revisar qué cursos exigen más lectura o trabajo en casa.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">5</div>
                      <span class="timeline-day">Día 5</span>
                      <span class="timeline-content">
                        Acercarte a <b>bienestar/psicopedagogía</b> para conocer horarios y opciones de atención.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">6</div>
                      <span class="timeline-day">Día 6</span>
                      <span class="timeline-content">
                        Realizar un bloque de estudio completo (lectura + resumen + preguntas) 
                        y ajustar horarios según cómo te sentiste en la semana.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">7</div>
                      <span class="timeline-day">Día 7</span>
                      <span class="timeline-content">
                        Revisar la semana: logros académicos, ánimo general y qué funcionó mejor. 
                        Anotar 2–3 cosas a mantener la próxima semana.
                      </span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Mensaje final
            st.markdown(
            """
            <div class="card">
              <h5>Mensaje final</h5>
              <p>
              Tu promedio excelente muestra que <b>ya tienes la capacidad académica</b>. 
              Ahora el reto es alinear tu día a día (asistencia, tareas, organización y cuidado emocional) con ese potencial.
              </p>
              <p>
              No estás fallando; estás en una etapa exigente donde se juntan trabajo, estudios, economía y emociones. 
              Con pequeños ajustes sostenidos y apoyo adecuado, es totalmente posible mantenerte en la carrera y crecer en tu vocación.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        elif caso_sel["Nombre"] == "Aylan Mostacero":
        # 1. Diagnóstico del estado actual
            st.markdown(
            """
            <div class="card">
              <h5>Diagnóstico del estado actual</h5>
              <p><b>Perfil general:</b></p>
              <p>
              Estudias <b>Teología</b>, tienes <b>30 años</b> (adulto). 
              Tu promedio ponderado es <b>bueno</b>, lo que indica capacidades académicas sólidas. 
              Trabajas a <b>tiempo completo</b>, con <b>asistencia baja</b> y <b>desarrollo de tareas bajo</b>.
              </p>
              <p><b>Contexto familiar y económico:</b></p>
              <p>
              Procedes de <b>provincia</b> y vives con <b>familiares</b>, lo que implica apoyo pero también responsabilidades. 
              Tienes <b>deuda financiera media</b>. Tu padre está <b>desempleado</b> y tu madre es <b>ama de casa</b>, 
              por lo que probablemente sientas una fuerte responsabilidad económica.
              </p>
              <p><b>Estado emocional:</b></p>
              <p>
              Presentas <b>sintomatología depresiva en nivel medio</b>, que puede afectar energía, motivación, 
              concentración y esperanza respecto al futuro académico.
              </p>
              <p>
              <b>Conclusión inicial:</b> hay una tensión fuerte entre tu deseo de estudiar y la realidad: 
              trabajo a tiempo completo, presión económica, responsabilidad familiar y estado emocional. 
              Esto explica por qué, aunque tu promedio es bueno, tu asistencia y tareas están bajas.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Nivel de riesgo alto
            st.markdown(
            """
            <div class="card">
              <h5>Nivel de riesgo de deserción: ALTO</h5>
              <p>Se explica principalmente por la combinación de:</p>
              <ul>
                <li><b>Económico:</b> deuda media + padre desempleado + trabajo a tiempo completo → tendencia a priorizar el trabajo.</li>
                <li><b>Académico:</b> asistencia baja y pocas tareas → desconexión con clases y vacíos en contenidos.</li>
                <li><b>Emocional:</b> síntomas depresivos medios → cansancio, apatía, pensamientos como “no voy a poder con todo”.</li>
                <li><b>Familiar:</b> posible carga de “tengo que responder por mi familia” y poca comprensión de la exigencia universitaria.</li>
              </ul>
              <p>
              No estás en riesgo alto por falta de capacidad, sino por <b>sobrecarga de responsabilidades + contexto económico + estado emocional</b>.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Fortalezas y debilidades
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                """
                <div class="card">
                  <h5>Fortalezas</h5>
                  <ul>
                    <li><b>Promedio bueno</b> a pesar de trabajar tiempo completo, tener deuda y síntomas depresivos.</li>
                    <li>Estar en Teología implica <b>interés por el sentido, los valores y el servicio</b>.</li>
                    <li>Sigues matriculado → a pesar de todo, <b>no has renunciado</b> a tu proyecto.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with col2:
                st.markdown(
                """
                <div class="card">
                  <h5>Debilidades / puntos críticos</h5>
                  <ul>
                    <li>Dificultad para equilibrar <b>trabajo tiempo completo + estudios</b>.</li>
                    <li><b>Baja asistencia</b> + pocas tareas → si continúa, tu promedio caerá y aumentará la frustración.</li>
                    <li><b>Presión económica y familiar</b> que puede llevar a pausar o abandonar la carrera.</li>
                    <li><b>Síntomas depresivos</b> que bajan energía y empujan a postergar y desconectarte.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 4. Recomendaciones + 5. Plan 7 días
            col_rec, col_plan = st.columns(2)

            with col_rec:
                st.markdown(
                """
                <div class="card">
                  <h5>Recomendaciones personalizadas</h5>
                  <span class="pill">Académicas</span>
                  <ul>
                    <li><b>Revisión urgente de carga académica:</b> evaluar reducir créditos/cursos si trabajas tiempo completo.</li>
                    <li>Definir una <b>estrategia mínima de asistencia</b>: no faltar a clases críticas salvo emergencia real.</li>
                    <li>Plan de tareas <b>“modo supervivencia”</b>: priorizar lo que más pesa en la nota y consolida contenidos clave.</li>
                    <li>Técnica rápida de estudio: leer 10–15 minutos, escribir 3 ideas clave y 1 pregunta por texto.</li>
                  </ul>
                  <span class="pill">Emocionales</span>
                  <ul>
                    <li>Reconocer que estás bajo <b>mucha presión</b> (no es debilidad sentirte sobrepasado).</li>
                    <li>Buscar <b>apoyo psicológico o de consejería</b> si la universidad lo ofrece.</li>
                    <li>Rutina diaria de autocuidado (10–15 min): respiración, caminata breve, escribir qué te pesó y qué agradeces.</li>
                  </ul>
                  <span class="pill">Económicas / familiares</span>
                  <ul>
                    <li>Obtener <b>claridad sobre la deuda</b>: monto exacto, opciones de fraccionamiento, fechas límite reales.</li>
                    <li>Explorar <b>beneficios o becas</b> internas por situación económica.</li>
                    <li>Conversar con la familia, en la medida de lo posible, explicando que estás en un punto delicado.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with col_plan:
                st.markdown(
                """
                <div class="card">
                  <h5>Plan de acción: próximos 7 días</h5>
                  <div class="timeline">
                    <div class="timeline-item">
                      <div class="timeline-circle">1</div>
                      <span class="timeline-day">Días 1–2</span>
                      <span class="timeline-content">
                        Hacer una lista de cursos y horarios de trabajo. 
                        Marcar en un calendario trabajo, clases y 2–3 bloques de estudio (30–45 min) reales.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">2</div>
                      <span class="timeline-day">Día 3</span>
                      <span class="timeline-content">
                        Escribir o acudir a coordinación/tutor académico para revisar tu carga de cursos y priorizar materias.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">3</div>
                      <span class="timeline-day">Día 4</span>
                      <span class="timeline-content">
                        Ir a oficina de créditos/deuda y pedir información clara sobre monto, plazos y opciones.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">4</div>
                      <span class="timeline-day">Día 5</span>
                      <span class="timeline-content">
                        Acercarte a bienestar universitario / servicio psicológico para pedir orientación por estrés y presión económica.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">5</div>
                      <span class="timeline-day">Día 6</span>
                      <span class="timeline-content">
                        Cumplir al menos <b>un bloque de estudio</b> y entregar una tarea pendiente, aunque no esté perfecta.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">6</div>
                      <span class="timeline-day">Día 7</span>
                      <span class="timeline-content">
                        Revisar la semana: asistencia, tareas y pasos para pedir ayuda. 
                        Ajustar el plan para la siguiente semana.
                      </span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 6. Mensaje final
            st.markdown(
            """
            <div class="card">
              <h5>Mensaje final</h5>
              <p>
              Tu situación es objetivamente difícil: no es falta de ganas, es una combinación de trabajo, economía, familia y estado emocional.
              </p>
              <p>
              Pero hay dos datos clave: tu <b>promedio es bueno</b> y <b>sigues matriculado</b>. 
              Eso muestra capacidad y perseverancia. El riesgo alto no significa que todo esté decidido,
              sino que es urgente <b>hacer cambios y pedir ayuda</b>, no que todo esté perdido.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        elif caso_sel["Nombre"] == "Esau Morales":
        # 1. Diagnóstico del estado actual
            st.markdown(
            """
            <div class="card">
              <h5>Diagnóstico del estado actual</h5>
              <p><b>Perfil general:</b></p>
              <p>
              Estudias en <b>Ciencias de la Salud</b>, tienes <b>30 años</b> (adulto).
              Tu promedio ponderado es <b>bueno</b>, lo que indica buenas bases y capacidad académica.
              No trabajas actualmente, lo que te da más tiempo para estudiar, aunque puede aumentar la preocupación económica.
              </p>
              <p>
              Tu <b>asistencia es media</b> y el <b>desarrollo de tareas es alto</b>: cumples muy bien con trabajos e informes, 
              pero no siempre estás presente en clase.
              </p>
              <p><b>Contexto personal y familiar:</b></p>
              <p>
              Procedes de <b>provincia</b> y has tenido un <b>traslado</b>, lo que implica adaptación a nueva ciudad y entorno.
              Vives con <b>familiares</b> en un contexto donde tu padre está <b>desempleado</b> y tu madre es <b>ama de casa</b>, 
              ambos con nivel educativo de <b>secundaria</b>.
              </p>
              <p><b>Estado emocional:</b></p>
              <p>
              Presentas <b>sintomatología depresiva baja</b>: señales leves, pero importantes de cuidar (cansancio, preocupación, desánimo ocasional).
              </p>
              <p>
              <b>Conclusión inicial:</b> académicamente estás bien orientado, pero hay puntos de atención en asistencia, 
              presión económica y adaptación familiar/emocional. Esto sustenta un nivel de riesgo <b>medio</b>.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Fortalezas y debilidades
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                """
                <div class="card">
                  <h5>Fortalezas académicas</h5>
                  <ul>
                    <li><b>Promedio bueno</b> y <b>tareas altas</b> → responsabilidad y organización con trabajos e informes.</li>
                    <li>En Ciencias de la Salud esto es clave por la carga práctica y de reportes.</li>
                    <li><b>No trabajas</b> actualmente → tienes posibilidad real de organizar mejor tu tiempo de estudio.</li>
                    <li><b>30 años</b> → mayor claridad de propósito y responsabilidad sobre tu vocación.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with col2:
                st.markdown(
                """
                <div class="card">
                  <h5>Debilidades / puntos de alerta</h5>
                  <ul>
                    <li><b>Asistencia media</b> en una carrera donde las clases prácticas y laboratorios son vitales.</li>
                    <li><b>Factor económico:</b> deuda media + padre desempleado + madre ama de casa → presión por “no fallar”.</li>
                    <li><b>Traslado desde provincia</b> → adaptación a nuevo entorno, posibles expectativas altas de la familia.</li>
                    <li><b>Apoyo académico limitado</b> en casa (padres con secundaria).</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3–4. Recomendaciones + Plan 7 días
            col_rec, col_plan = st.columns(2)

            with col_rec:
                st.markdown(
                """
                <div class="card">
                  <h5>Recomendaciones personalizadas</h5>
                  <span class="pill">Académicas</span>
                  <ul>
                    <li>Tratar la <b>asistencia</b> como algo “no negociable” en cursos clave (prácticas, laboratorios).</li>
                    <li>Usar tu fortaleza en tareas para <b>reforzar contenidos de clase</b>, no solo para “entregar”.</li>
                    <li>Diseñar una <b>semana tipo</b> con bloques fijos de estudio y repaso (1–2 h al día).</li>
                    <li>Hablar con un <b>docente o tutor</b> para identificar cursos/competencias críticos.</li>
                  </ul>
                  <span class="pill">Emocionales</span>
                  <ul>
                    <li>Aunque los síntomas sean <b>bajos</b>, revisarlos a diario con 3 preguntas: ánimo (1–10), preocupación, gratitud.</li>
                    <li>Buscar un <b>espacio de conversación segura</b> (familiar, amigo, consejería, psicología).</li>
                    <li>Programar <b>descanso intencional</b> sin culpa: 15–20 min para caminar, respirar, despejarte.</li>
                  </ul>
                  <span class="pill">Económicas / familiares</span>
                  <ul>
                    <li>Obtener <b>claridad sobre la deuda</b> (monto, fechas, fraccionamiento).</li>
                    <li>Explorar <b>becas o apoyos internos</b> (por situación económica o por rendimiento).</li>
                    <li>Conversar con la familia sobre lo que implica estudiar Ciencias de la Salud y tus esfuerzos actuales.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with col_plan:
                st.markdown(
                """
                <div class="card">
                  <h5>Plan de acción: próximos 7 días</h5>
                  <div class="timeline">
                    <div class="timeline-item">
                      <div class="timeline-circle">1</div>
                      <span class="timeline-day">Día 1</span>
                      <span class="timeline-content">
                        Hacer tu calendario semanal con clases, horas de estudio y momentos de descanso.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">2</div>
                      <span class="timeline-day">Día 2</span>
                      <span class="timeline-content">
                        Ir o escribir a administración para aclarar tu deuda (cifras y fechas).
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">3</div>
                      <span class="timeline-day">Día 3</span>
                      <span class="timeline-content">
                        Hacer una lista de cursos más difíciles o importantes y decidir a cuáles no puedes faltar.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">4</div>
                      <span class="timeline-day">Día 4</span>
                      <span class="timeline-content">
                        Acercarte a un docente de confianza o tutor académico para contar tu situación y pedir orientación.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">5</div>
                      <span class="timeline-day">Día 5</span>
                      <span class="timeline-content">
                        Hacer un <b>repaso corto</b> el mismo día de una clase clave (20–30 min) y registrar cómo te sentiste (1–10).
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">6</div>
                      <span class="timeline-day">Día 6</span>
                      <span class="timeline-content">
                        Conversar con algún familiar sobre tus avances y preguntar por apoyo de bienestar estudiantil/psicológico.
                      </span>
                    </div>
                    <div class="timeline-item">
                      <div class="timeline-circle">7</div>
                      <span class="timeline-day">Día 7</span>
                      <span class="timeline-content">
                        Revisar la semana: asistencia, orden en estudio, claridad económica. 
                        Anotar 2 cosas que funcionaron bien para repetir la próxima semana.
                      </span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 6. Mensaje final
            st.markdown(
            """
            <div class="card">
              <h5>Mensaje final</h5>
              <p>
              Tu perfil muestra capacidades académicas sólidas (<b>promedio bueno</b>) y alta responsabilidad en tareas, 
              en un contexto económico-familiar complejo.
              </p>
              <p>
              El riesgo es <b>medio</b>, pero sin carga laboral externa este es un muy buen momento para ordenarte, 
              pedir apoyo y consolidar tu camino en Ciencias de la Salud. 
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    
# ==========================
# Vista Clustering
# ==========================
def vista_clustering():
    st.title("📊 Vista de Clustering")

    # --- Estilos para badges ---
    st.markdown(
        """
        <style>
        .cluster-badges {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        .badge {
            padding: 0.6rem 1rem;
            border-radius: 999px;
            color: white;
            font-size: 0.9rem;
            font-weight: 600;
            box-shadow: 0 0 8px rgba(0,0,0,0.25);
        }
        .badge-1 {
            background: linear-gradient(135deg, #e53935, #ff7043);  /* rojo */
        }
        .badge-2 {
            background: linear-gradient(135deg, #1e88e5, #42a5f5);  /* azul */
        }
        .badge-3 {
            background: linear-gradient(135deg, #6a1b9a, #ab47bc);  /* morado */
        }
        .badge-title {
            font-weight: 700;
            margin-right: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Badges de interpretación ---
    st.markdown(
        """
        <div class="cluster-badges">
            <div class="badge badge-1">
                <span class="badge-title">Cluster 1:</span>
                Deuda alta, rendimiento medio
            </div>
            <div class="badge badge-2">
                <span class="badge-title">Cluster 2:</span>
                Buen rendimiento, alta presión
            </div>
            <div class="badge badge-3">
                <span class="badge-title">Cluster 3:</span>
                Sobrecarga académica con dificultad de adaptación
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Tabla ---
    DATA_PATH = BASE_DIR / "DATA-CLUSTERING-SOLO-NUMERO.xlsx"

    try:
        df_cluster = pd.read_excel(DATA_PATH)

        # estilos de tabla
        st.markdown(
        """
        <style>
        /* Contenedor redondeado */
        .stDataFrame div[data-testid="stHorizontalBlock"] {
        border-radius: 12px;
        overflow: hidden;
        }

        /* Quitar líneas fuertes y dejar look plano */
        .stDataFrame table {
        border-collapse: collapse;
        font-size: 0.9rem;
        border: none;
        }
        .stDataFrame td, .stDataFrame th {
        border: none !important;
        padding: 0.5rem 0.75rem;
        }

        /* Header oscuro suave */
        .stDataFrame th {
        background: #111827;
        color: #e5e7eb;
        font-weight: 600;
        }

        /* Fila hover sutil */
        .stDataFrame tbody tr:hover {
        background-color: rgba(15, 23, 42, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True
        )

        st.data_editor(
        df_cluster,
        use_container_width=True,
        disabled=True,          # solo lectura
        )

    except Exception as e:
        st.error(f"Error cargando el Excel de clustering:\n{e}")


# ==========================
# Selector de vista
# ==========================
vista = st.radio(
    "Selecciona vista",
    ["Redes Neuronales", "Clustering"],
    horizontal=True
)

if vista == "Redes Neuronales":
    vista_redes()
else:
    vista_clustering()