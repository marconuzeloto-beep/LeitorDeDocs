"""
Integração com Ollama para extração de documentos via Llama local.
"""
import json
import logging
import urllib.error
import urllib.request

from openai import OpenAI

logger = logging.getLogger(__name__)

# Modelos com suporte a visão, em ordem de preferência
MODELOS_VISAO = [
    "llama3.2-vision",
    "llava",
    "minicpm-v",
    "moondream",
]


class OllamaIndisponivel(Exception):
    """Levantada quando o Ollama não está acessível ou o modelo não está instalado."""
    pass


class OllamaProvider:
    """Executa modelos Llama localmente via Ollama."""

    def __init__(self, base_url: str, model: str):
        self._client = OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key="ollama",  # valor obrigatório pela lib, ignorado pelo Ollama
        )
        self._model = model

    @property
    def nome(self) -> str:
        return f"Ollama ({self._model})"

    def gerar_conteudo(self, prompt: str, imagens_base64: list[str]) -> str:
        conteudo = [{"type": "text", "text": prompt}]
        for img_b64 in imagens_base64:
            conteudo.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        resposta = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": conteudo}],
            temperature=0.1,
            max_tokens=2048,
        )
        return resposta.choices[0].message.content.strip()


def conectar(base_url: str, modelo: str) -> OllamaProvider:
    """
    Verifica a conexão com o Ollama, detecta o modelo e retorna o provider.
    Levanta OllamaIndisponivel com instruções claras em caso de erro.
    """
    # 1. Verifica se o servidor está acessível
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
    except urllib.error.URLError:
        raise OllamaIndisponivel(
            f"Ollama não está acessível em {base_url}.\n"
            "  → Instale em: https://ollama.com\n"
            "  → Inicie com: ollama serve"
        )

    # 2. Lista modelos instalados
    instalados = [m["name"] for m in data.get("models", [])]
    logger.info(f"Modelos Ollama instalados: {instalados}")

    if not instalados:
        raise OllamaIndisponivel(
            "Nenhum modelo instalado no Ollama.\n"
            f"  → Execute: ollama pull {modelo}"
        )

    # 3. Encontra o modelo a usar
    modelo_escolhido = _escolher_modelo(modelo, instalados)
    logger.info(f"Modelo selecionado: {modelo_escolhido}")
    return OllamaProvider(base_url=base_url, model=modelo_escolhido)


def _escolher_modelo(preferido: str, instalados: list[str]) -> str:
    """Retorna o modelo a usar: o preferido se disponível, senão o melhor instalado."""
    bases_instaladas = [n.split(":")[0] for n in instalados]

    # Tenta usar o modelo configurado
    if preferido:
        if preferido in instalados:
            return preferido
        base = preferido.split(":")[0]
        if base in bases_instaladas:
            return next(n for n in instalados if n.startswith(base))
        # Modelo pedido não está instalado — avisa e tenta fallback
        logger.warning(
            f"Modelo '{preferido}' não está instalado.\n"
            f"  → Para instalar: ollama pull {preferido}\n"
            f"  Tentando usar alternativa de visão instalada..."
        )

    # Seleciona o melhor modelo de visão entre os instalados
    for pref in MODELOS_VISAO:
        base = pref.split(":")[0]
        if pref in instalados or base in bases_instaladas:
            return next((n for n in instalados if n.startswith(base)), pref)

    # Nenhum modelo de visão encontrado
    sugerido = preferido or MODELOS_VISAO[0]
    raise OllamaIndisponivel(
        f"Nenhum modelo de visão encontrado entre os instalados: {instalados}\n"
        f"  → Execute: ollama pull {sugerido}"
    )
