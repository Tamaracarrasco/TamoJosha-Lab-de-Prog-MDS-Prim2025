# Conclusiones

## Aprendizaje y reflexión sobre el proceso MLOps

Trabajar en este proyecto me permitió experimentar de primera mano lo que significa llevar un modelo de machine learning desde el desarrollo hasta la producción. No fue solo entrenar un modelo y obtener métricas, fue entender todo el ecosistema que rodea a un sistema de ML en producción. Lidiar casi que en todos los programas como Docker Desktop, VSCode, Airflow, MLFlow, con problemáticas, errores, resolver situaciones, etc. que realmente me ponían a prueba. Hubo momentos en que de verdad no sabía cuál era el problema y así fui encontrando aspectos importantes a tener en consideración que en el fondo aprendí bien, y que podré saber cómo sortear en entregas o trabajos futuros.

## Tracking

Una de las cosas que más valoré fue tener un sistema de tracking desde el inicio. Poder comparar experimentos, ver qué combinaciones de hiperparámetros funcionaban mejor y tener un registro histórico de todo lo que intenté me dio una claridad que no tendría si solo estuviera guardando scripts sueltos. Esto me hizo más disciplinada en mi forma de experimentar, porque sabía que cada corrida quedaba documentada.

Me di cuenta de cuánto tiempo perdía antes tratando de recordar qué configuración había usado en un experimento anterior. Con el tracking automático, esa fricción desapareció completamente.

Además, por ejemplo los warnings o los logs que uno puede definir para saber dónde ver que hay errores o secciones donde quizás no se está leyendo o escuchando el puerto, poder setear eso con logs fue clave. Poner como mensajes también en los cumplimientos de etapas y que en el fondo se vieran esos mensajes, también era bueno y hacía mi trabajo más eficiente y dirigido de alguna manera.

## Despliegue

El despliegue fue probablemente la parte más desafiante desde el punto de vista de algo nuevo por decir, ya que una cosa es tener un modelo que funciona en un Jupyter notebook y otra muy distinta exponerlo como un servicio que otras personas pueden usar.

Trabajar con Gradio fue interesante porque me permitió crear una interfaz e interactuar quizas con librerias y funciones que no conocia, pero también me hizo pensar en aspectos que nunca había considerado como el hecho de ¿qué pasa si alguien sube un archivo mal? o ¿cómo manejo los errores de forma clara para el usuario? ¿qué información del modelo debo mostrar y qué es mejor ocultar? Me sirvió un poco también para poner alertas o ciertas cosas como "informativas" dentro de la app para que el usuario sepa bien qué está haciendo mal, cómo ayudarlo "a distancia" por decir, porque uno no siempre va a estar ahí al lado para ayudar a las personas a que esto funcione.

El desafío más grande fue optimizar los tiempos de respuesta. Nuestro dataset de cerca de 9 millones de registros no corre muy rápido, tuve problemas con el tema de Optuna también, por lo que optamos como por hacer etapas de ambientes en el desarrollo e incluso en un momento fijar parametros (de la entrega 1) porque no habia forma de que convergiera el modelo. Actualmente, solo correrá con todos los datos en una etapa de producción, pero por ahora corre con menos porque el costo computacional es alto. Eso en el fondo es algo que uno siempre debe tener en mente, porque al final si las cosas no corren o se acaba la RAM, etc., da lo mismo si tienes un buen modelo si el caso es que no puedes correrlo o sacar un resultado.

## Airflow y la orquestación de pipelines

Airflow me cambió la perspectiva sobre cómo debería estructurar los flujos de trabajo. Antes pensaba en scripts que se ejecutaban uno tras otro, pero con Airflow empecé a pensar en términos de dependencias, recuperación ante fallos y modularidad. Y esto es muy bueno, eficiente, intuitivo, te avisa las cosas con los logs, te muestra los gráficos, los tiempos de correr de cada etapa, te muestra si hay errores, etc. Muy práctico.

Lo más valioso fue entender que un pipeline productivo no es solo "que funcione", sino que sea resiliente. Si falla un paso, ¿puedo reintentar solo esa parte? ¿Puedo monitorear dónde está el cuello de botella? Airflow me dio esas capacidades y me hizo pensar más como un ingeniero de datos y menos como alguien que solo experimenta con modelos.

También me di cuenta de que la documentación del flujo es tan importante como el código mismo. Cuando volví a revisar los DAGs después de unos días, agradecí haber escrito descripciones claras de qué hacía cada tarea.

También los README son cosas clave, me sirven harto sobre todo para dejar ahí el tema de los comandos con los que corro el docker-compose y para que eso levante todos mis scripts. El orden de las carpetas también, que cada una sea intuitiva, dejar todo bien registrado, con nombres claros, no redundantes. Que se ponga título también en los scripts con la semi ruta del archivo me sirve también para ubicarme localmente porque a veces mirar las carpetas es un poco abrumador cuando son muchas.

## Lo que mejoraría en el futuro

Si tuviera que iterar sobre este proyecto, hay varias cosas que agregaría:

Primero, implementaría un sistema de monitoreo más robusto en producción. Me gustaría saber no solo si el modelo está funcionando, sino cómo se está comportando, por ejemplo, en los tiempos de inferencia, distribución de las predicciones, posibles cambios en los datos de entrada. Creo que esto es crucial para detectar degradación del modelo tempranamente.

También automatizaría más el proceso de reentrenamiento. Actualmente tengo que activar manualmente el pipeline cuando quiero reentrenar, pero sería ideal tener un sistema que detecte cuándo el rendimiento cae por debajo de cierto umbral y active automáticamente un ciclo de reentrenamiento con datos frescos.

En cuanto a métricas, me gustaría expandir más allá de la precisión básica. Agregaría métricas de negocio que realmente importen en el contexto del problema, y establecería alertas cuando estas métricas se desvíen de lo esperado. Por que igual es importante integrar este aspecto en terminos estrategicos y de negocio, como que no deberiamos poder evaluar estos resultados solos sin este complemento.

## Reflexión final

Este proyecto me hizo apreciar que MLOps no es solo un conjunto de herramientas, sino una mentalidad. Es pensar en el ciclo de vida completo del modelo, desde la primera línea de código hasta el mantenimiento en producción. Es ser consciente de que el modelo es solo una pieza de un sistema más grande que incluye datos, infraestructura, monitoreo y procesos.

Lo que más me llevo es la importancia de la iteración. Ningún sistema sale perfecto a la primera, y está bien. Lo importante es tener la infraestructura adecuada para poder mejorar continuamente sin romper lo que ya funciona.