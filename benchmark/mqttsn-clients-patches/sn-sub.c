/* sn-sub.c
 *
 * PATCHED VERSION for libcoap-pqc-bench benchmark integration
 * Added timing measurement hooks for benchmark data collection
 *
 * Copyright (C) 2024 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
 *
 * Copyright (C) 2006-2024 wolfSSL Inc.
 *
 * This file is part of wolfMQTT.
 *
 * wolfMQTT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * wolfMQTT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1335, USA
 */

/* Include the autoconf generated config.h */
#ifdef HAVE_CONFIG_H
    #include <config.h>
#endif

#include "wolfmqtt/mqtt_client.h"

#include "sn-client.h"
#include "mqttnet.h"

/* ============================================================================
 * BENCHMARK TIMING HOOKS (added for libcoap-pqc-bench integration)
 * Mirrors the approach used in libcoap-patches/coap-client.c
 * ============================================================================ */
#include <time.h>
#include <stdint.h>

/* Function to get the current time in nanoseconds */
static uint64_t get_current_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Function to append timing data to the benchmark output file */
static void append_time_to_file(unsigned long long total_time_ns) {
    /* Get the repository root from environment */
    char *repo_root = getenv("REPO_ROOT");
    /* Check for BENCH_DATA_DIR environment variable first (set by benchmark scripts) */
    char *bench_data_dir = getenv("BENCH_DATA_DIR");
    char filepath[512];

    if (bench_data_dir != NULL) {
        /* Use BENCH_DATA_DIR if set (allows flexible output directory) */
        snprintf(filepath, sizeof(filepath), "%s/time_output.txt", bench_data_dir);
    } else if (repo_root != NULL) {
        /* Fallback to default data/current directory */
        snprintf(filepath, sizeof(filepath), "%s/benchmark/data/current/time_output.txt", repo_root);
    } else {
        /* Fallback to current directory */
        strcpy(filepath, "time_output.txt");
    }

    FILE *file = fopen(filepath, "a");  /* Open in append mode */
    if (file != NULL) {
        fprintf(file, "%.3f\n", total_time_ns / 1000000000.0);
        fclose(file);
    } else {
        perror("Failed to open time output file");
    }
}
/* ============================================================================ */

/* Locals */
static int mStopRead = 0;

#ifdef WOLFMQTT_SN

/* Configuration */
/* Maximum size for network read/write callbacks. */
#ifndef MAX_BUFFER_SIZE
#define MAX_BUFFER_SIZE 1024
#endif
#define TEST_MESSAGE    "test"
#define SHORT_TOPIC_NAME "s1"

static int sn_message_cb(MqttClient *client, MqttMessage *msg,
    byte msg_new, byte msg_done)
{
    byte buf[PRINT_BUFFER_SIZE+1];
    word32 len;
    word16 topicId;
    MQTTCtx* mqttCtx = (MQTTCtx*)client->ctx;

    if (msg_new) {
        if (!(msg->topic_type & SN_TOPIC_ID_TYPE_SHORT)) {
            /* Topic ID name */
            topicId = (word16)((byte)msg->topic_name[0] << 8 |
                               (byte)msg->topic_name[1]);

            /* Print incoming message */
            PRINTF("MQTT-SN Message: Topic ID %hu, Qos %d, Id %d, Len %u",
                    topicId, msg->qos, msg->packet_id, msg->total_len);
        }
        else {
            /* Topic short name */
            /* Print incoming message */
            PRINTF("MQTT-SN Message: Topic ID %c%c, Qos %d, Id %d, Len %u",
                    msg->topic_name[0], msg->topic_name[1],
                    msg->qos, msg->packet_id, msg->total_len);
        }
        /* for test mode: check if TEST_MESSAGE was received */
        if (mqttCtx != NULL && mqttCtx->test_mode) {
            if (XSTRLEN(TEST_MESSAGE) == msg->buffer_len &&
                XSTRNCMP(TEST_MESSAGE, (char*)msg->buffer,
                         msg->buffer_len) == 0)
            {
                mStopRead = 1;
            }
        }
    }

    /* Print message payload */
    len = msg->buffer_len;
    if (len > PRINT_BUFFER_SIZE) {
        len = PRINT_BUFFER_SIZE;
    }
    XMEMCPY(buf, msg->buffer, len);
    buf[len] = '\0'; /* Make sure its null terminated */
    PRINTF("........Payload (%d - %d): %s",
        msg->buffer_pos, msg->buffer_pos + len, buf);

    if (msg_done) {
        PRINTF("....MQTT-SN Message: Done");
    }

    /* Return negative to terminate publish processing */
    return MQTT_CODE_SUCCESS;
}

/* The Register callback is used when the gateway
   assigns a new topic ID to a topic name. */
static int sn_reg_callback(word16 topicId, const char* topicName, void *ctx)
{
    PRINTF("MQTT-SN Register CB: New topic ID: %hu : \"%s\"", topicId, topicName);
    (void)ctx;

    return(MQTT_CODE_SUCCESS);
}

#ifdef WOLFMQTT_DISCONNECT_CB
/* callback indicates a network error or broker disconnect occurred */
static int mqtt_disconnect_cb(MqttClient* client, int error_code, void* ctx)
{
    (void)client;
    (void)ctx;
    PRINTF("Disconnect Callback: %s (error %d)",
        MqttClient_ReturnCodeToString(error_code), error_code);
    return 0;
}
#endif

int sn_test(MQTTCtx *mqttCtx)
{
    int rc = MQTT_CODE_SUCCESS;
    word16 topicID;

    /* BENCHMARK: Start timing measurement */
    uint64_t start_time = get_current_time_ns();

    PRINTF("MQTT-SN Client: Client ID %s, QoS %d",
            mqttCtx->client_id,
            mqttCtx->qos);

    /* Initialize Network */
    rc = SN_ClientNet_Init(&mqttCtx->net, mqttCtx);
    PRINTF("MQTT-SN Net Init: %s (%d)",
        MqttClient_ReturnCodeToString(rc), rc);
    if (rc != MQTT_CODE_SUCCESS) {
        goto exit;
    }

    /* setup tx/rx buffers */
    mqttCtx->tx_buf = (byte*)WOLFMQTT_MALLOC(MAX_BUFFER_SIZE);
    mqttCtx->rx_buf = (byte*)WOLFMQTT_MALLOC(MAX_BUFFER_SIZE);

    /* Initialize MqttClient structure */
    rc = MqttClient_Init(&mqttCtx->client, &mqttCtx->net,
        sn_message_cb,
        mqttCtx->tx_buf, MAX_BUFFER_SIZE,
        mqttCtx->rx_buf, MAX_BUFFER_SIZE,
        mqttCtx->cmd_timeout_ms);

    PRINTF("MQTT-SN Init: %s (%d)",
        MqttClient_ReturnCodeToString(rc), rc);
    if (rc != MQTT_CODE_SUCCESS) {
        goto exit;
    }

    /* The client.ctx will be stored in the cert callback ctx during
       MqttSocket_Connect for use by mqtt_tls_verify_cb */
    mqttCtx->client.ctx = mqttCtx;

#if defined(ENABLE_MQTT_TLS) && defined(WOLFSSL_DTLS)
    if (mqttCtx->use_tls) {
        /* Set the DTLS flag in the client structure to indicate DTLS usage */
        MqttClient_Flags(&mqttCtx->client, 0, MQTT_CLIENT_FLAG_IS_DTLS);
    }
#endif

    /* Setup socket direct to gateway */
    rc = MqttClient_NetConnect(&mqttCtx->client, mqttCtx->host,
           mqttCtx->port, DEFAULT_CON_TIMEOUT_MS,
           mqttCtx->use_tls, mqtt_dtls_cb);

    PRINTF("MQTT-SN Socket Connect: %s (%d)",
        MqttClient_ReturnCodeToString(rc), rc);
    if (rc != MQTT_CODE_SUCCESS) {
        goto exit;
    }

    /* Set the Register callback used when the gateway
       assigns a new topic ID to a topic name. */
    rc = SN_Client_SetRegisterCallback(&mqttCtx->client, sn_reg_callback, NULL);
    if (rc != MQTT_CODE_SUCCESS) {
        goto exit;
    }

#ifdef WOLFMQTT_DISCONNECT_CB
    /* setup disconnect callback */
    rc = MqttClient_SetDisconnectCallback(&mqttCtx->client,
        mqtt_disconnect_cb, NULL);
    if (rc != MQTT_CODE_SUCCESS) {
        goto exit;
    }
#endif

    {
        SN_Connect connect_s, *connect = &connect_s;
        /* Build connect packet */
        XMEMSET(connect, 0, sizeof(SN_Connect));
        connect->keep_alive_sec = mqttCtx->keep_alive_sec;
        connect->clean_session = mqttCtx->clean_session;
        connect->client_id = mqttCtx->client_id;
        connect->protocol_level = SN_PROTOCOL_ID;

        /* Last will and testament sent by broker to subscribers
            of topic when broker connection is lost */
        connect->enable_lwt = mqttCtx->enable_lwt;
        if (connect->enable_lwt) {
            /* Send client id in LWT payload */
            connect->will.qos = mqttCtx->qos;
            connect->will.retain = 0;
            connect->will.willTopic = WOLFMQTT_TOPIC_NAME"lwttopic";
            connect->will.willMsg = (byte*)mqttCtx->client_id;
            connect->will.willMsgLen =
              (word16)XSTRLEN(mqttCtx->client_id);
        }

        PRINTF("MQTT-SN Connect: gateway = %s : %d",
                mqttCtx->host, mqttCtx->port);
        /* Send Connect and wait for Connect Ack */
        rc = SN_Client_Connect(&mqttCtx->client, connect);

        if (rc != MQTT_CODE_SUCCESS) {
            PRINTF("MQTT-SN Connect: %s (%d)",
                MqttClient_ReturnCodeToString(rc), rc);
            goto disconn;
        }

        /* Validate Connect Ack info */
        PRINTF("....MQTT-SN Connect Ack: Return Code %u",
                connect->ack.return_code);
    }

    /* Either the register or the subscribe block could be used to get the
       topic ID. Both are done here as an example of using the API. */
    {
        /* Register topic name to get the assigned topic ID */
        SN_Register regist_s, *regist = &regist_s;

        XMEMSET(regist, 0, sizeof(SN_Register));
        regist->packet_id = mqtt_get_packetid();
        regist->topicName = DEFAULT_TOPIC_NAME;

        PRINTF("MQTT-SN Register: topic = %s", regist->topicName);
        rc = SN_Client_Register(&mqttCtx->client, regist);

        if ((rc == 0) && (regist->regack.return_code == SN_RC_ACCEPTED)) {
            /* Topic ID is returned in RegAck */
            topicID = regist->regack.topicId;
        }
        PRINTF("....MQTT-SN Register Ack: rc = %d, topic id = %hu",
                regist->regack.return_code, regist->regack.topicId);
    }

    {
        /* Subscribe Topic */
        SN_Subscribe subscribe;

        XMEMSET(&subscribe, 0, sizeof(SN_Subscribe));

        subscribe.duplicate = 0;
        subscribe.qos = MQTT_QOS_0;
        subscribe.topic_type = SN_TOPIC_ID_TYPE_NORMAL;
        subscribe.topicNameId = DEFAULT_TOPIC_NAME;
        subscribe.packet_id = mqtt_get_packetid();

        PRINTF("MQTT-SN Subscribe: topic name = %s", subscribe.topicNameId);
        rc = SN_Client_Subscribe(&mqttCtx->client, &subscribe);

        PRINTF("....MQTT-SN Subscribe Ack: topic id = %hu, rc = %d",
                subscribe.subAck.topicId, subscribe.subAck.return_code);

        if ((rc == 0) && (subscribe.subAck.return_code == SN_RC_ACCEPTED)) {
            /* Topic ID is returned in SubAck */
            topicID = subscribe.subAck.topicId;
        }
        goto disconn;
    }

    /* Read Loop - commented out for benchmark (single subscribe only) */
    /*
    PRINTF("MQTT Waiting for message...");

    do {
        if (mStopRead) {
            rc = MQTT_CODE_SUCCESS;
            PRINTF("MQTT Exiting...");
            break;
        }

        rc = SN_Client_WaitMessage(&mqttCtx->client,
                                   mqttCtx->cmd_timeout_ms);

        if (rc == MQTT_CODE_ERROR_TIMEOUT) {
            PRINTF("Keep-alive timeout, sending ping");
            rc = SN_Client_Ping(&mqttCtx->client, NULL);
            if (rc != MQTT_CODE_SUCCESS) {
                PRINTF("MQTT-SN Ping Keep Alive Error: %s (rc = %d)",
                    MqttClient_ReturnCodeToString(rc), rc);
                break;
            }
        }
        else if (rc != MQTT_CODE_SUCCESS) {
            PRINTF("MQTT-SN Message Wait Error: %s (rc = %d)",
                MqttClient_ReturnCodeToString(rc), rc);
            break;
        }
    } while (1);
    */

disconn:
    /* Disconnect */
    rc = SN_Client_Disconnect(&mqttCtx->client);

    PRINTF("MQTT Disconnect: %s (rc = %d)",
        MqttClient_ReturnCodeToString(rc), rc);
    if (rc != MQTT_CODE_SUCCESS) {
        goto disconn;
    }

    rc = MqttClient_NetDisconnect(&mqttCtx->client);

    PRINTF("MQTT Socket Disconnect: %s (rc = %d)",
        MqttClient_ReturnCodeToString(rc), rc);

exit:

    /* BENCHMARK: Stop timing and record result */
    {
        uint64_t end_time = get_current_time_ns();
        uint64_t elapsed_ns = end_time - start_time;
        append_time_to_file(elapsed_ns);
    }

    /* Free resources */
    if (mqttCtx->tx_buf) WOLFMQTT_FREE(mqttCtx->tx_buf);
    if (mqttCtx->rx_buf) WOLFMQTT_FREE(mqttCtx->rx_buf);

    /* Cleanup network */
    MqttClientNet_DeInit(&mqttCtx->net);

    MqttClient_DeInit(&mqttCtx->client);

    return rc;
}

#endif /* WOLFMQTT_SN */

/* so overall tests can pull in test function */
    #ifdef USE_WINDOWS_API
        #include <windows.h> /* for ctrl handler */

        static BOOL CtrlHandler(DWORD fdwCtrlType)
        {
            if (fdwCtrlType == CTRL_C_EVENT) {
                mStopRead = 1;
                PRINTF("Received Ctrl+c");
                return TRUE;
            }
            return FALSE;
        }
    #elif HAVE_SIGNAL
        #include <signal.h>
        static void sig_handler(int signo)
        {
            if (signo == SIGINT) {
                mStopRead = 1;
                PRINTF("Received SIGINT");
            }
        }
    #endif

#if defined(NO_MAIN_DRIVER)
int sn_main(int argc, char** argv)
#else
int main(int argc, char** argv)
#endif
{
    int rc;
#ifdef WOLFMQTT_SN
    MQTTCtx mqttCtx;

    /* init defaults */
    mqtt_init_ctx(&mqttCtx);
    mqttCtx.app_name = "sn-client";
    mqttCtx.client_id = DEFAULT_CLIENT_ID"-SN";

    /* Settings for MQTT-SN gateway */
    mqttCtx.host = "localhost";
    mqttCtx.port = 10000;

    /* parse arguments */
    rc = mqtt_parse_args(&mqttCtx, argc, argv);
    if (rc != 0) {
        return rc;
    }
#endif
#ifdef USE_WINDOWS_API
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler,
          TRUE) == FALSE)
    {
        PRINTF("Error setting Ctrl Handler! Error %d", (int)GetLastError());
    }
#elif HAVE_SIGNAL
    if (signal(SIGINT, sig_handler) == SIG_ERR) {
        PRINTF("Can't catch SIGINT");
    }
#endif

#ifdef WOLFMQTT_SN
    rc = sn_test(&mqttCtx);
#else
    (void)argc;
    (void)argv;

    /* This example requires MQTT-SN mode to be enabled
       ./configure --enable-sn */
    PRINTF("Example not compiled in!");
    rc = EXIT_FAILURE;
#endif


    return (rc == MQTT_CODE_SUCCESS) ? 0 : EXIT_FAILURE;
}

